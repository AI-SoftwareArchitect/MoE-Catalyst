from torch import optim
from config.base_config import device
import torch
import torch.nn as nn
import time

# =============================================================================
# ADVANCED TRAINING
# =============================================================================

def train_model(model, batches, epochs=50, lr=1e-4):
    """Enhanced training with advanced optimizer and progress tracking"""

    if not batches:
        print("❌ No batches available for training!")
        return

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    model.train()
    best_loss = float('inf')
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0
        num_batches = 0
        batch_count = 0
        batch_len = len(batches)

        for batch_input, batch_target in batches:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            if batch_count % 10 == 0:
                print(f"Batch {batch_count}/{batch_len}")

            batch_count += 1

            optimizer.zero_grad()
            logits = model(batch_input)
            logits = logits.view(-1, logits.size(-1))
            targets = batch_target.view(-1)

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

        # Süre ve tahmin hesaplama
        elapsed_time = time.time() - start_time
        epoch_time = time.time() - epoch_start
        remaining_time = (epochs - epoch - 1) * epoch_time
        percent_complete = (epoch + 1) / epochs * 100

        # Formatlama
        def format_time(seconds):
            mins, secs = divmod(int(seconds), 60)
            hrs, mins = divmod(mins, 60)
            return f"{hrs}h {mins}m {secs}s"

        print(
            f"🧠 Epoch {epoch + 1}/{epochs} "
            f"| ⏱️ Elapsed: {format_time(elapsed_time)} "
            f"| 📉 Loss: {avg_loss:.4f} "
            f"| 🔁 LR: {optimizer.param_groups[0]['lr']:.6f} "
            f"| ✅ {percent_complete:.1f}% "
            f"| ⏳ ETA: {format_time(remaining_time)}"
        )