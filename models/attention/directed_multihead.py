from typing import Optional
import torch.nn as nn
import math
import torch
from torch.nn import functional as F
# =============================================================================
# ADVANCED MULTIHEAD ATTENTION
# =============================================================================
class DirectedMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, sparse_topk=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.sparse_topk = sparse_topk

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.head_gates = nn.Parameter(torch.ones(num_heads))
        self.relative_pos_bias = nn.Parameter(torch.zeros((2 * 512 - 1), num_heads))

    def _generate_relative_positions(self, seq_len, device):
        range_vec = torch.arange(seq_len, device=device)
        rel_pos = range_vec[None, :] - range_vec[:, None] + 511
        return rel_pos

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        rel_pos_idx = self._generate_relative_positions(T, x.device)
        pos_bias = self.relative_pos_bias[rel_pos_idx]
        pos_bias = pos_bias.permute(2, 0, 1).unsqueeze(0)
        attn_scores = attn_scores + pos_bias

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        if self.sparse_topk > 0:
            topk = torch.topk(attn_scores, self.sparse_topk, dim=-1)
            topk_mask = torch.full_like(attn_scores, float('-inf'))
            topk_mask.scatter_(-1, topk.indices, topk.values)
            attn_scores = topk_mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        gated_weights = attn_weights * self.head_gates.view(1, self.num_heads, 1, 1)

        attn_output = torch.matmul(gated_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(attn_output)