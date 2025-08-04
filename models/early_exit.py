from typing import Tuple
import torch.nn as nn
import torch
# =============================================================================
# EARLY EXIT CLASSIFIER
# =============================================================================
class EarlyExitClassifier(nn.Module):
    """Confidence-based early exit mechanism"""

    def __init__(self, embed_dim: int, vocab_size: int, threshold: float = 0.9):
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        logits = self.classifier(x)  # [B, seq_len, vocab_size]
        confidence = self.confidence_head(x).mean()
        should_exit = confidence.item() > self.threshold
        return logits, should_exit
