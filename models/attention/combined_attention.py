# combined_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedAttention(nn.Module):
    """Normalize edilmiş fusion weights, fusion MLP, LayerNorm"""
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        # logits to be normalized via softmax
        self.stm_logit = nn.Parameter(torch.tensor(0.0))
        self.ltm_logit = nn.Parameter(torch.tensor(0.0))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, stm_output: torch.Tensor, ltm_output: torch.Tensor) -> torch.Tensor:
        # stm_output, ltm_output: (B, T, D)
        assert stm_output.shape == ltm_output.shape, "STM/LTM shape mismatch"
        w = torch.stack([self.stm_logit, self.ltm_logit], dim=0)
        w = F.softmax(w, dim=0)
        # fused channels
        combined = torch.cat([w[0] * stm_output, w[1] * ltm_output], dim=-1)  # (B,T,2D)
        out = self.fusion(combined)
        return self.norm(out)
