from torch.utils.data import Dataset
import torch
import re
import numpy as np
# =============================================================================
# TEXT DATASET
# =============================================================================
class TextDataset(Dataset):
    def __init__(self, token_file_path, seq_len):
        self.seq_len = seq_len
        print("Token'lar yükleniyor...")
        self.all_tokens_tensor = torch.from_numpy(np.load(token_file_path)).long()
        self.indices = [(i, i + self.seq_len) for i in range(len(self.all_tokens_tensor) - self.seq_len)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx, end_idx = self.indices[idx]
        input_seq = self.all_tokens_tensor[start_idx:end_idx]
        target_seq = self.all_tokens_tensor[start_idx + 1:end_idx + 1]
        return input_seq, target_seq