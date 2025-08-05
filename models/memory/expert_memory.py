import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertMemory(nn.Module):
    def __init__(self, embed_dim, num_experts=4, expert_hidden_dim=256, top_k=1, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.top_k = top_k

        # Experts: her biri küçük bir feedforward ağ
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, expert_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden_dim, embed_dim)
            )
            for _ in range(num_experts)
        ])

        # Gating mekanizması: girdi embed_dim'ye göre uzmanları seçer
        self.gate = TopKGate(embed_dim, num_experts, top_k)

    def forward(self, x):
        """
        x: (B, T, embed_dim)
        Returns:
            output: (B, T, embed_dim)
        """
        B, T, C = x.size()
        x_flat = x.view(B * T, C)  # (B*T, embed_dim)

        # Gating skorları ve seçilen uzmanlar
        gates, expert_indices = self.gate(x_flat)  # gates: (B*T, top_k), expert_indices: (B*T, top_k)

        # Experts çıktılarını hesapla
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(x_flat)  # (B*T, embed_dim)
            expert_outputs.append(expert_out.unsqueeze(1))  # (B*T, 1, embed_dim)
        expert_outputs = torch.cat(expert_outputs, dim=1)  # (B*T, num_experts, embed_dim)

        # Seçilen uzman çıktılarını gate ile çarp ve topla
        mask = torch.zeros_like(expert_outputs)  # (B*T, num_experts, embed_dim)
        for k in range(self.top_k):
            idx = expert_indices[:, k]
            gate_vals = gates[:, k].unsqueeze(1)  # (B*T,1)
            mask[torch.arange(B * T), idx] = gate_vals
        output = (expert_outputs * mask).sum(dim=1)  # (B*T, embed_dim)

        return output.view(B, T, C)
