from torch import optim
from config.base_config import device
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from monitoring.base_monitoring import TrainingMonitor


# =============================================================================
# ADVANCED TRAINING
# =============================================================================
def train_model(model, batches, epochs=1, lr=5e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for x, y in batches:
            # x, y CPU’dan gelebilir; doğru cihaza taşı
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)  # labels forward'a veriliyorsa uygun şekilde y ekleyin
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()