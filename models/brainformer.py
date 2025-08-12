# brainformer.py
import torch
import torch.nn as nn

from config.base_config import BrainFormerConfig
from models.attention.combined_attention import CombinedAttention
from models.dual_brain import DualBrainTransformer
from models.early_exit import EarlyExitClassifier
from models.memory.long_term import LongTermMemory
from models.memory.short_term import ShortTermMemory
from models.moe.moe_layer import MoELayer


# ... mevcut importlar


class BrainFormer(nn.Module):
    def __init__(self, config: BrainFormerConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        self.dual_brain = DualBrainTransformer(config)

        self.stm = ShortTermMemory(config.d_model, memory_size=config.memory_size)
        self.ltm = LongTermMemory(config.d_model, buffer_size=config.ltm_buffer_size)
        self.combined_attention = CombinedAttention(config.d_model)

        self.early_exit = EarlyExitClassifier(config.d_model, config.vocab_size, threshold=config.early_exit_threshold)

        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)

        self._init_weights()

        # MoE bayrağı ve hiperparametreleri (varsayılanlarla)
        self.use_moe = getattr(config, "use_moe", False)
        self.moe_lambda = getattr(config, "moe_lambda", 1e-2)

        if self.use_moe:
            self.moe = MoELayer(
                embed_dim=config.d_model,
                num_experts=getattr(config, "num_experts", 4),
                expert_hidden_dim=getattr(config, "expert_hidden_dim", config.ff_hidden_dim),
                top_k=getattr(config, "top_k", 1),
                dropout=getattr(config, "dropout", 0.1),
                expert_dropout=getattr(config, "expert_dropout", 0.1),
            )
        else:
            self.moe = None

        self.last_aux_loss = None  # eğitim döngüsünde okunacak

        # ... lm_head ve diğer katmanlar

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def _create_causal_mask(self, seq_len: int, device: torch.device):
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1,1,seq,seq)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Embedding ağırlığının cihazını referans al
        device = self.token_embedding.weight.device

        # Girdiyi doğru cihaza ve doğru tipe taşı
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        input_ids = input_ids.to(device)

        # Pozisyon indeksleri aynı cihazda üretilmeli
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(input_ids.size(0), -1)

        # Varsa attention_mask CPU’da gelmiş olabilir; aynı cihaza taşıyın
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        token_emb = self.token_embedding(input_ids)
        pos_emb = self.pos_embedding(position_ids)

        x = token_emb + pos_emb

        # Eğer causal mask veya benzeri mask üretimi varsa, onun da device=device ile oluştuğundan emin olun
        causal_mask = self._create_causal_mask(seq_len, device=device)  # _create_causal_mask içinde device kullanın

        # Devam eden forward hesapları ...

        left_out, right_out = self.dual_brain(x, causal_mask)

        # Memory integration
        stm_out = self.stm(left_out)  # (B, T, D)
        ltm_out = self.ltm(right_out, store=self.training, retrieve=False)  # (B, T, D)

        combined = self.combined_attention(stm_out, ltm_out)  # (B, T, D)

        early_logits, should_exit = self.early_exit(combined)  # logits: (B,T,V), should_exit: (B,)
        # if any batch element wants exit and eval mode, return per-example early logits
        if not self.training:
            if should_exit.all().item():
                return early_logits
            # else continue for those who didn't meet threshold (advanced: per-sample routing)
        final_output = self.final_norm(combined)
        
        # ... embedding + transformer blokları
        hidden = final_output  # (B, T, D) - mevcut son gizli durum

        aux_loss = None
        if self.moe is not None:
            B, T, D = hidden.shape
            moe_out, aux_loss = self.moe(hidden.reshape(B * T, D))
            hidden = hidden + moe_out.view(B, T, D)  # residual ekleme

        self.last_aux_loss = aux_loss  # torch skalar veya None

        logits = self.output_proj(hidden)
        return logits