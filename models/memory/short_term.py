# short_term_memory.py
import torch
import torch.nn as nn

class ShortTermMemory(nn.Module):
    """
    Simpler, robust STM:
    - keeps a learnable small memory bank
    - integrates via gating and returns (B, T, D) shaped output (no hard-assert)
    """
    def __init__(self, embed_dim: int, memory_size: int = 16, device: torch.device = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_size = memory_size
        self.device = device or torch.device("cpu")

        self.memory = nn.Parameter(torch.randn(1, memory_size, embed_dim) * 0.02)

        self.input_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

        self.forget_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, memory_size),
            nn.Sigmoid()
        )

        # compressor: reduce concatenated (T+M) feature per-token via projection
        self.project = nn.Linear((memory_size + 1) * embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        mem = self.memory.expand(B, -1, -1)  # (B, M, D)

        input_g = self.input_gate(x)  # (B, T, 1)
        gated_inputs = x * input_g  # (B, T, D)

        pooled = x.mean(dim=1)  # (B, D)
        forget = self.forget_gate(pooled).unsqueeze(-1)  # (B, M, 1)
        mem = mem * (1 - forget)

        # For each token, concatenate token + flattened memory and project back
        mem_flat = mem.view(B, -1)  # (B, M*D)
        mem_rep = mem_flat.unsqueeze(1).expand(-1, T, -1)  # (B, T, M*D)
        token_and_mem = torch.cat([gated_inputs, mem_rep], dim=-1)  # (B, T, D + M*D)
        # project back to (B, T, D) — uses larger projection
        # To keep sizes stable, we compress with a linear that expects (D + M*D)
        # For safety, use a linear with input dim (D + M*D)
        out = self.project(token_and_mem)  # (B, T, D)
        return out
