# topk_gate.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class TopKGate(nn.Module):
    def __init__(self, embed_dim: int, num_experts: int, top_k: int = 1):
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.linear = nn.Linear(embed_dim, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (N, C)
        returns:
          gates: (N, top_k)  (softmax over topk logits)
          indices: (N, top_k) long
        """
        logits = self.linear(x)  # (N, E)
        topk_logits, topk_indices = torch.topk(logits, k=self.top_k, dim=-1)  # (N, K)
        gates = F.softmax(topk_logits, dim=-1)  # (N, K)
        return gates, topk_indices
