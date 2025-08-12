from torch.utils.data import Dataset
import torch
import numpy as np
from pathlib import Path

class TextDataset(Dataset):
    def __init__(self, token_file_path, seq_len):
        self.seq_len = int(seq_len)
        print("Token'lar yükleniyor...")

        path = Path(token_file_path)
        if not path.exists():
            raise FileNotFoundError(f"Token dosyası bulunamadı: {path.resolve()}")

        try:
            arr = np.load(path, allow_pickle=False, mmap_mode='r')
        except ValueError as e:
            # Genellikle object array yükleme hatası
            raise ValueError(
                "tokens.npy bir object array olabilir. Lütfen token'ları sayısal (np.int64) olarak kaydedin."
            ) from e

        # 1D hale getir
        if arr.ndim != 1:
            arr = arr.reshape(-1)

        # Tamsayıya çevir
        if arr.dtype.kind not in ("i", "u"):
            try:
                arr = arr.astype(np.int64)
            except Exception as e:
                raise TypeError(f"Beklenen tamsayı token dizisi, alınan dtype={arr.dtype}") from e

        tokens = arr.astype(np.int64, copy=False)
        self.all_tokens_tensor = torch.from_numpy(tokens)

        n = len(self.all_tokens_tensor)
        if n <= self.seq_len:
            raise ValueError(
                f"Yetersiz token sayısı: {n}. seq_len={self.seq_len} için en az {self.seq_len + 1} token gerekli."
            )

        self.indices = [(i, i + self.seq_len) for i in range(n - self.seq_len)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx, end_idx = self.indices[idx]
        input_seq = self.all_tokens_tensor[start_idx:end_idx]
        target_seq = self.all_tokens_tensor[start_idx + 1:end_idx + 1]
        return input_seq, target_seq