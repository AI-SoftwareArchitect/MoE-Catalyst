import torch
import torch.nn as nn
# =============================================================================
# ADVANCED MEMORY SYSTEMS -=SHORT TERM=-
# =============================================================================
class ShortTermMemory(nn.Module):
    """Enhanced STM with gating mechanism"""

    def __init__(self, embed_dim: int, memory_size: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_size = memory_size

        # Learnable memory bank
        self.memory = nn.Parameter(torch.randn(1, memory_size, embed_dim) * 0.1)

        # Gate mechanism
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Expand memory for batch
        memory = self.memory.expand(B, -1, -1)

        # Gate-controlled memory integration
        gate_scores = self.gate(x)  # (B, T, 1)
        gated_x = x * gate_scores

        # Combine with memory
        combined = torch.cat([gated_x, memory], dim=1)
        return combined