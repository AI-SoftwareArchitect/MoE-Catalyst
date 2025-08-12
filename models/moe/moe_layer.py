import torch
import torch.nn as nn

from models.moe.topk_gate import TopKGate


class MoELayer(nn.Module):
    def __init__(self, embed_dim: int, num_experts: int = 4, expert_hidden_dim: int = 2048, top_k: int = 2, dropout: float = 0.1, expert_dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_dropout = nn.Dropout(expert_dropout)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, expert_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden_dim, embed_dim),
            ) for _ in range(num_experts)
        ])
        self.gate = TopKGate(embed_dim, num_experts, top_k)

    def forward(self, x: torch.Tensor):
        """
        x: (N, D)
        returns: (N, D), aux_loss (load balancing)
        """
        gates, idxs = self.gate(x)  # gates: (N,K), idxs: (N,K)
        N, D = x.shape

        # Efficient batch per expert
        out = torch.zeros_like(x)
        aux_loss = 0

        for expert_id in range(self.num_experts):
            # Mask positions for this expert
            mask = (idxs == expert_id)  # (N, top_k)
            if not mask.any():
                continue

            # Positions and weights for this expert
            positions = mask.any(dim=1).nonzero(as_tuple=True)[0]  # (M,)
            if positions.numel() == 0:
                continue

            # Collect inputs and gates for expert
            expert_gates = torch.zeros(positions.size(0), device=x.device, dtype=x.dtype)
            for k in range(self.top_k):
                expert_gates += gates[positions, k] * mask[positions, k].to(x.dtype)

            expert_inputs = x[positions]  # (M, D)
            expert_inputs = self.expert_dropout(expert_inputs)

            # Forward expert in batch
            expert_outputs = self.experts[expert_id](expert_inputs)  # (M, D)

            # Weighted sum outputs
            out[positions] += expert_outputs * expert_gates.unsqueeze(-1)

            # Load balancing loss (coefficient 1e-2, can be tuned)
            prob_expert = gates[:, 0].mean()  # avg gate prob for expert at top_k=1
            aux_loss += prob_expert * prob_expert  # penalize imbalance (can be refined)

        return out, aux_loss
