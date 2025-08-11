# long_term.py
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

DROPOUT_DEFAULT: float = 0.1
DECAY_RATE_DEFAULT: float = 0.995


class LongTermMemory(nn.Module):
    """
    Basit uzun süreli bellek (LTM) bileşeni:
      - store: Önem skoruna göre dairesel tampona yazma
      - retrieve: Normalize edilmiş kosinüs benzerliği ile top-k vektörleri ağırlıklı toplama
    """

    def __init__(
            self,
            embed_dim: int,
            buffer_size: int = 4096,
            device: Optional[torch.device] = None,
            decay_rate: float = DECAY_RATE_DEFAULT,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.buffer_size = int(buffer_size)
        self.device = device or torch.device("cpu")
        self.decay_rate = float(decay_rate)

        # Önem skoru üreticisi (0..1)
        self.importance_scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),
        )
        # Geri uyumluluk için alias (Renaming)
        self.importance_gate = self.importance_scorer

        # İsteğe bağlı dönüştürücü (şu an kullanılmıyor ama korunuyor)
        self.memory_transform = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(DROPOUT_DEFAULT),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # Dairesel tampon ve göstergeleri
        self.register_buffer("ltm_buffer", torch.zeros(self.buffer_size, embed_dim))
        self.register_buffer("ltm_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("ltm_full", torch.tensor(False, dtype=torch.bool))

        # Modülü hedef cihaza taşı
        self.to(self.device)

    # ------ Helper functions (Extract Function) ------

    @staticmethod
    def _normalize(t: torch.Tensor) -> torch.Tensor:
        return F.normalize(t, dim=-1, eps=1e-6)

    def _buffer_length(self) -> int:
        if bool(self.ltm_full.item()):
            return self.buffer_size
        return int(self.ltm_ptr.item())

    @torch.no_grad()
    def _compute_store_mask(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.importance_scorer(x).squeeze(-1)  # (N,)
        threshold = scores.mean()
        return scores > threshold  # (N,)

    @torch.no_grad()
    def _ring_buffer_write(self, items: torch.Tensor) -> None:
        """
        items: (n, D), cihazı ltm_buffer ile hizalanmış olmalı.
        """
        if items.numel() == 0:
            return
        ptr = int(self.ltm_ptr.item())
        n = int(items.size(0))
        end = (ptr + n) % self.buffer_size

        if ptr + n <= self.buffer_size:
            self.ltm_buffer[ptr:ptr + n] = items
        else:
            first = self.buffer_size - ptr
            self.ltm_buffer[ptr:] = items[:first]
            self.ltm_buffer[:end] = items[first:]

        # Tampon doluluğunu güncelle
        if not bool(self.ltm_full.item()) and (ptr + n) >= self.buffer_size:
            self.ltm_full.fill_(True)

        # Pointer'ı güvenle güncelle (register_buffer korunur)
        self.ltm_ptr.fill_(end)

        # Global çürüme uygula
        self.ltm_buffer.mul_(self.decay_rate)

    # ------ Public API ------

    @torch.no_grad()
    def store(self, x: torch.Tensor) -> None:
        """
        x: (N, D) - yazmaya aday düzleştirilmiş temsiller
        """
        if x.dim() != 2 or x.size(-1) != self.embed_dim:
            raise ValueError(f"store expects (N, {self.embed_dim}) got {tuple(x.shape)}")

        if self.buffer_size == 0:
            return

        x = x.to(self.ltm_buffer.device)
        mask = self._compute_store_mask(x)
        items = x[mask]
        if items.numel() == 0:
            return

        self._ring_buffer_write(items)

    def retrieve(self, x: torch.Tensor, top_k: int = 16) -> torch.Tensor:
        """
        x: (B, T, D)
        Returns: (B, T, D) - bellekten top-k yumuşak toplama
        """
        if x.dim() != 3 or x.size(-1) != self.embed_dim:
            raise ValueError(f"retrieve expects (B, T, {self.embed_dim}) got {tuple(x.shape)}")

        length = self._buffer_length()
        if length == 0:
            return torch.zeros_like(x)

        mems = self.ltm_buffer[:length].to(x.device)  # (M, D)
        mems_n = self._normalize(mems)
        queries = self._normalize(x)  # (B, T, D)

        # Kosinüs skorları ve top-k seçim
        scores = torch.einsum("btc,mc->btm", queries, mems_n)  # (B, T, M)
        k = max(1, min(int(top_k), length))
        topk_scores, topk_idx = torch.topk(scores, k=k, dim=-1)  # (B, T, k)
        weights = F.softmax(topk_scores, dim=-1)  # (B, T, k)

        # Seçilen bellek vektörlerini al ve ağırlıklı topla
        selected = mems[topk_idx]  # (B, T, k, D)
        weighted = selected * weights.unsqueeze(-1)  # (B, T, k, D)
        return weighted.sum(dim=2)  # (B, T, D)

    def forward(self, x: torch.Tensor, store: bool = False, retrieve: bool = False, top_k: int = 16) -> torch.Tensor:
        """
        Basit yönlendirme:
          - store=True ise x (N, D) beklenir ve bellek güncellenir; çıktı x döndürülür.
          - retrieve=True ise x (B, T, D) beklenir ve bellekten alınan temsil döndürülür.
          - Her ikisi de False ise x aynen döndürülür.
        """
        if store and retrieve:
            raise ValueError("store and retrieve aynı anda True olamaz.")

        if store:
            store_x = x if x.dim() == 2 else x.reshape(-1, x.size(-1))
            self.store(store_x)
            return x

        if retrieve:
            return self.retrieve(x, top_k=top_k)

        return x
