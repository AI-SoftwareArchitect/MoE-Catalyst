# transformer_block.py
import torch
import torch.nn as nn
from typing import Optional

from models.attention.directed_multihead import DirectedMultiheadAttention


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.1, max_pos: int = 512):
        super().__init__()
        self.attention = DirectedMultiheadAttention(embed_dim, num_heads, dropout, sparse_topk=0, max_pos=max_pos)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attention(self.norm1(x), mask)
        x = x + attn_out
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        return x
