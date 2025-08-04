import torch.nn as nn
import torch
class CombinedAttention(nn.Module):
    """Fusion mechanism for STM and LTM outputs"""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        # Learnable fusion weights
        self.stm_weight = nn.Parameter(torch.ones(1))
        self.ltm_weight = nn.Parameter(torch.ones(1))

    def forward(self, stm_output: torch.Tensor, ltm_output: torch.Tensor) -> torch.Tensor:
        # Weighted combination
        weighted_stm = self.stm_weight * stm_output
        weighted_ltm = self.ltm_weight * ltm_output

        # Concatenate and fuse
        combined = torch.cat([weighted_stm, weighted_ltm], dim=-1)
        return self.fusion(combined)