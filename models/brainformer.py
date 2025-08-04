from torch import nn
import torch
from config.base_config import BrainFormerConfig
from models.attention.combined_attention import CombinedAttention
from models.dual_brain import DualBrainTransformer
from models.early_exit import EarlyExitClassifier
from models.memory.long_term import LongTermMemory
from models.memory.short_term import ShortTermMemory

# =============================================================================
# MAIN BRAINFORMER MODEL
# =============================================================================
class BrainFormer(nn.Module):
    """Advanced Brain-like Transformer Architecture"""

    def __init__(self, config: BrainFormerConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # Dual brain architecture
        self.dual_brain = DualBrainTransformer(config)

        # Memory systems
        self.stm = ShortTermMemory(config.d_model, config.memory_size)
        self.ltm = LongTermMemory(config.d_model)
        self.combined_attention = CombinedAttention(config.d_model)

        # Early exit mechanism
        self.early_exit = EarlyExitClassifier(config.d_model, config.vocab_size, config.early_exit_threshold)

        # Output layers
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def _create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal attention mask"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos_ids)
        x = token_emb + pos_emb

        # Create causal mask
        mask = self._create_causal_mask(seq_len).to(input_ids.device)

        # Dual brain processing
        left_output, right_output = self.dual_brain(x, mask)

        # Memory integration (her token için)
        stm_out = self.stm(left_output)  # [B, seq_len+memory_size, d_model]
        ltm_out = self.ltm(right_output, store=self.training)  # [B, seq_len, d_model]

        # Sadece ilk seq_len token'ı kullan (STM ve LTM'nin ilk seq_len kısmı)
        stm_out = stm_out[:, :seq_len, :]  # [B, seq_len, d_model]
        ltm_out = ltm_out[:, :seq_len, :]  # [B, seq_len, d_model]

        # Combine memory outputs (her token için)
        combined = self.combined_attention(stm_out, ltm_out)  # [B, seq_len, d_model]

        # Early exit check (her token için)
        early_logits, should_exit = self.early_exit(combined)  # [B, seq_len, vocab_size]
        if should_exit and not self.training:
            return early_logits  # [B, seq_len, vocab_size]

        # Final processing
        final_output = self.final_norm(combined)
        logits = self.output_proj(final_output)  # [B, seq_len, vocab_size]
        return logits