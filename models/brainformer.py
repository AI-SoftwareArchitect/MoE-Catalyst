# brainformer.py
import torch
import torch.nn as nn

from config.base_config import BrainFormerConfig
from models.attention.combined_attention import CombinedAttention
from models.dual_brain import DualBrainTransformer
from models.early_exit import EarlyExitClassifier
from models.memory.long_term import LongTermMemory
from models.memory.short_term import ShortTermMemory


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

    def forward(self, input_ids: torch.LongTensor):
        device = input_ids.device
        B, seq_len = input_ids.shape

        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos_ids)
        x = token_emb + pos_emb

        mask = self._create_causal_mask(seq_len, device=device)

        left_out, right_out = self.dual_brain(x, mask)

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
        logits = self.output_proj(final_output)
        return logits
