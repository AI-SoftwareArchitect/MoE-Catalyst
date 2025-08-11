# early_exit.py
import torch
import torch.nn as nn
from typing import Tuple

class EarlyExitClassifier(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int, threshold: float = 0.95):
        super().__init__()
        self.threshold = threshold
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, vocab_size)
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.BoolTensor]:
        # x: (B, T, D)
        logits = self.classifier(x)  # (B,T,V)
        conf = self.confidence_head(x).squeeze(-1)  # (B,T)
        per_example_conf = conf.mean(dim=1)  # (B,)
        should_exit = per_example_conf > self.threshold  # (B,)
        return logits, should_exit
