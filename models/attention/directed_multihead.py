# directed_multihead.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional flash-attn detection (placeholder)
try:
    import flash_attn  # pragma: no cover
    HAS_FLASH = True
except Exception:
    HAS_FLASH = False

class DirectedMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, sparse_topk: int = 0, max_pos: int = 512):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.sparse_topk = sparse_topk
        self.max_pos = max_pos

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.head_gates = nn.Parameter(torch.ones(num_heads))
        self.relative_pos_bias = nn.Parameter(torch.zeros(2 * max_pos - 1, num_heads))

    def _rel_pos_bias(self, seq_len: int, device: torch.device):
        idx = torch.arange(seq_len, device=device)
        rel = idx[None, :] - idx[:, None] + (self.max_pos - 1)  # [0, 2*max_pos-2]
        # returns (seq_len, seq_len, num_heads)
        return self.relative_pos_bias[rel]  # (seq_len, seq_len, num_heads)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,T,Dh)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,T,T)

        # add relative bias
        pos_bias = self._rel_pos_bias(T, x.device).permute(2, 0, 1).unsqueeze(0)  # (1,H,T,T)
        attn_scores = attn_scores + pos_bias

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        if self.sparse_topk > 0:
            # keep only top-k per query position
            topk_vals, topk_idx = torch.topk(attn_scores, self.sparse_topk, dim=-1)  # (B,H,T,k)
            mask_full = torch.full_like(attn_scores, float('-inf'))  # (B,H,T,T)
            # scatter topk_vals into mask_full at positions topk_idx
            # prepare index for scatter: expand dims to match
            mask_full.scatter_(-1, topk_idx, topk_vals)
            attn_scores = mask_full

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        gated = attn_weights * self.head_gates.view(1, self.num_heads, 1, 1)
        attn_out = torch.matmul(gated, v)  # (B,H,T,Dh)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_out)
