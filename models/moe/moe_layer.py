import torch
import torch.nn as nn
import torch.nn.functional as F
from .topk_gate import TopKGate

class MoELayer(nn.Module):
    def __init__(self, embed_dim, num_experts, expert_hidden_dim, top_k=1, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Experts: her biri basit feedforward katmanlar olabilir
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, expert_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden_dim, embed_dim),
            ) for _ in range(num_experts)
        ])

        # Gate
        self.gate = TopKGate(embed_dim, num_experts, top_k)

    def forward(self, x):
        """
        x: (batch_size, embed_dim)
        """
        gates, expert_indices = self.gate(x)  # (B, top_k), (B, top_k)

        batch_size, embed_dim = x.size()
        top_k = self.top_k

        # Experts outputları için boş tensor
        expert_outputs = torch.zeros(batch_size, top_k, embed_dim, device=x.device, dtype=x.dtype)

        for k in range(top_k):
            indices = expert_indices[:, k]  # (B,)
            mask = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
            # Burada tüm batch için uzman bazlı işleme gitmek lazım. Kısaca batch içinde grupla:
            for expert_id in range(self.num_experts):
                idxs = (indices == expert_id).nonzero(as_tuple=True)[0]
                if idxs.numel() == 0:
                    continue
                expert_input = x[idxs]
                expert_output = self.experts[expert_id](expert_input)
                expert_outputs[idxs, k] = expert_output

        # Ağırlıklandır ve topla
        gates = gates.unsqueeze(-1)  # (B, top_k, 1)
        output = (expert_outputs * gates).sum(dim=1)  # (B, embed_dim)

        return output
