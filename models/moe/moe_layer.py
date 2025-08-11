# moe_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from topk_gate import TopKGate

class MoELayer(nn.Module):
    def __init__(self, embed_dim: int, num_experts: int = 4, expert_hidden_dim: int = 2048, top_k: int = 1, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, expert_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden_dim, embed_dim),
            ) for _ in range(num_experts)
        ])
        self.gate = TopKGate(embed_dim, num_experts, top_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, D)  N = batch * seq (flatten if necessary)
        returns: (N, D)
        """
        gates, idxs = self.gate(x)  # gates: (N,K), idxs: (N,K)
        N, D = x.shape
        out = x.new_zeros(N, D)

        # group positions per expert
        for expert_id in range(self.num_experts):
            # mask positions that route to this expert for any k
            mask_any = (idxs == expert_id).any(dim=1)  # (N,)
            if not mask_any.any():
                continue
            pos = mask_any.nonzero(as_tuple=True)[0]
            inp = x[pos]  # (M, D)
            expert_out = self.experts[expert_id](inp)  # (M, D)
            # compute weight per position: sum gates where idx == expert_id
            weight = x.new_zeros(pos.size(0))
            for k in range(self.top_k):
                sel = (idxs[pos, k] == expert_id).to(x.dtype) * gates[pos, k]
                weight = weight + sel
            out[pos] = out[pos] + expert_out * weight.unsqueeze(-1)
        return out
