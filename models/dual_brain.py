from typing import Optional
from torch import nn
import torch
from config.base_config import BrainFormerConfig
from models.transformer_block import TransformerBlock
from typing import Tuple
# =============================================================================
# DUAL BRAIN ARCHITECTURE
# =============================================================================
class DualBrainTransformer(nn.Module):
    """Left (Logic) and Right (Creative) brain processing"""

    def __init__(self, config: BrainFormerConfig):
        super().__init__()
        self.config = config

        # Left brain: Logical, structured processing
        self.left_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.num_heads, config.ff_hidden_dim, config.dropout)
            for _ in range(config.num_layers)
        ])

        # Right brain: Creative, noisy processing
        self.right_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.num_heads, config.ff_hidden_dim, config.dropout)
            for _ in range(config.num_layers)
        ])

        self.noise_scale = 0.1

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        left_x = x
        right_x = x

        # Process through both hemispheres
        for left_block, right_block in zip(self.left_blocks, self.right_blocks):
            # Left brain: structured processing
            left_x = left_block(left_x, mask)

            # Right brain: creative processing with noise injection
            noise = torch.randn_like(right_x) * self.noise_scale
            right_x = right_block(right_x + noise, mask)

        return left_x, right_x