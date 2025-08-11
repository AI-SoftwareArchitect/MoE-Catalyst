# dual_brain.py
import torch
import torch.nn as nn
from typing import Optional, Tuple
from models.transformer_block import TransformerBlock


class DualBrainTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.left_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.num_heads, config.ff_hidden_dim, config.dropout, max_pos=config.relative_max_pos)
            for _ in range(config.num_layers)
        ])
        self.right_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.num_heads, config.ff_hidden_dim, config.dropout, max_pos=config.relative_max_pos)
            for _ in range(config.num_layers)
        ])
        self.noise_scale = 0.05

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        left_x = x
        right_x = x
        for left_block, right_block in zip(self.left_blocks, self.right_blocks):
            left_x = left_block(left_x, mask)
            noise = torch.randn_like(right_x) * self.noise_scale
            right_x = right_block(right_x + noise, mask)
        return left_x, right_x
