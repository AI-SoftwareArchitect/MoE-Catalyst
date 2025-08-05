import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKGate(nn.Module):
    def __init__(self, embed_dim, num_experts, top_k=1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.linear = nn.Linear(embed_dim, num_experts)

    def forward(self, x):
        """
        x: (batch_size, embed_dim)
        Returns:
          gates: (batch_size, top_k) softmax ağırlıkları
          expert_indices: (batch_size, top_k) seçilen uzman indeksleri
        """
        logits = self.linear(x)  # (batch_size, num_experts)
        topk_logits, topk_indices = torch.topk(logits, self.top_k, dim=-1)  # en iyi top_k seç

        gates = F.softmax(topk_logits, dim=-1)  # normalize et

        return gates, topk_indices
