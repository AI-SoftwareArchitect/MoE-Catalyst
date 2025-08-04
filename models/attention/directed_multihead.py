from typing import Optional
import torch.nn as nn
import math
import torch
# =============================================================================
# ADVANCED MULTIHEAD ATTENTION
# =============================================================================
class DirectedMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.3):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Context bias for directed attention
        self.context_bias = nn.Parameter(torch.zeros(embed_dim))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better training"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                context_vector: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.size()
        H = self.num_heads
        head_dim = self.head_dim

        # Projections
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Context-directed attention
        if context_vector is not None:
            Q = Q + self.context_bias * context_vector

        # Reshape for multi-head attention
        Q = Q.view(B, T, H, head_dim).transpose(1, 2)  # (B, H, T, head_dim)
        K = K.view(B, T, H, head_dim).transpose(1, 2)
        V = V.view(B, T, H, head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)

        # Apply causal mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Attention weights and output
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)

        # Reshape back
        out = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)