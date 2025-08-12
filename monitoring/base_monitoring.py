# monitor.py
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

class TrainingMonitor:
    def __init__(self, log_dir="logs", use_tensorboard=True):
        self.use_tensorboard = use_tensorboard
        self.start_time = None
        self.batch_start = None
        self.global_step = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if use_tensorboard:
            self.writer = SummaryWriter(log_dir=log_dir)

        self.accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(self.device)

    def start_epoch(self):
        self.start_time = time.time()

    def start_batch(self):
        self.batch_start = time.time()

    def end_batch(self, loss, preds, targets):
        batch_time = time.time() - self.batch_start
        mem_alloc = torch.cuda.memory_allocated(self.device) / 1024 ** 2 if torch.cuda.is_available() else 0

        acc = self.accuracy_metric(preds, targets)

        if self.use_tensorboard:
            self.writer.add_scalar("Loss/train", loss.item(), self.global_step)
            self.writer.add_scalar("Accuracy/train", acc, self.global_step)
            self.writer.add_scalar("Perf/Batch_Time", batch_time, self.global_step)
            self.writer.add_scalar("Perf/GPU_Memory_MB", mem_alloc, self.global_step)

        self.global_step += 1

    def end_epoch(self, epoch):
        epoch_time = time.time() - self.start_time
        if self.use_tensorboard:
            self.writer.add_scalar("Perf/Epoch_Time", epoch_time, epoch)

    def close(self):
        if self.use_tensorboard:
            self.writer.close()
