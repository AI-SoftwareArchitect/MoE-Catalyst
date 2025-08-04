from typing import Optional
from torch import nn
import torch
from models.attention.directed_multihead import DirectedMultiheadAttention
# =============================================================================
# TRANSFORMER BLOCK
# =============================================================================
class TransformerBlock(nn.Module):
    """Enhanced transformer block with residual connections"""

    def __init__(self, embed_dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.3):
        super().__init__()

        self.attention = DirectedMultiheadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),  # Better activation than ReLU
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out = self.attention(self.norm1(x), mask)
        x = x + attn_out

        # Feed-forward with residual connection
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out

        return x