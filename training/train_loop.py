from torch import optim
from config.base_config import device
import torch

# =============================================================================
# ADVANCED TRAINING
# =============================================================================
def train_model(model, batches, epochs=50, lr=1e-4):
    """Enhanced training with advanced optimizer and scheduler"""
    if not batches:
        print("❌ No batches available for training!")
        return

    # Advanced optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999))

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    model.train()
    best_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for batch_input, batch_target in batches:
            batch_input = batch_input.to(device)  # [B, seq_len]
            batch_target = batch_target.to(device)  # [B, seq_len]

            optimizer.zero_grad()

            logits = model(batch_input)  # [B, seq_len, vocab_size]
            # CrossEntropyLoss expects [N, C] and targets [N]
            logits = logits.view(-1, logits.size(-1))  # [(B*seq_len), vocab_size]
            targets = batch_target.view(-1)  # [(B*seq_len)]

            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        scheduler.step(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
