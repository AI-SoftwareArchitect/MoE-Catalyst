import torch
import torch.nn as nn

# =============================================================================
# ADVANCED MEMORY SYSTEMS -=SHORT TERM=-
# =============================================================================
class ShortTermMemory(nn.Module):
    """Enhanced STM with gated update, learnable forget and memory compression"""

    def __init__(self, embed_dim: int, memory_size: int = 16, max_seq_len: int = 16, device=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_size = memory_size
        self.seq_len = max_seq_len
        self.device = device or torch.device("cpu")

        # Initialize learnable memory bank
        self.memory = nn.Parameter(torch.randn(1, memory_size, embed_dim) * 0.1)

        # Gating mechanism for input integration
        self.input_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

        # Forget gate to selectively forget memory contents
        self.forget_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, memory_size),
            nn.Sigmoid()
        )

        # Compression layer with fixed input size
        self.memory_compress = nn.Sequential(
            nn.Linear((self.seq_len + memory_size) * embed_dim, (self.seq_len + memory_size) * embed_dim // 2),
            nn.ReLU(),
            nn.Linear((self.seq_len + memory_size) * embed_dim // 2, (self.seq_len + memory_size) * embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Input tensor (B, T, C)
        Returns combined tensor (B, T + memory_size, C)
        """

        B, T, C = x.shape
        assert C == self.embed_dim, f"Embedding dimension mismatch: expected {self.embed_dim}, got {C}"
        assert T == self.seq_len, f"Expected sequence length {self.seq_len}, got {T}"

        # Expand memory for batch
        memory = self.memory.expand(B, -1, -1)  # (B, memory_size, embed_dim)

        # Compute input gate scores (B, T, 1)
        input_gates = self.input_gate(x)

        # Apply gating to inputs
        gated_inputs = x * input_gates

        # Compute forget gates for memory (B, memory_size, 1)
        pooled_x = x.mean(dim=1)  # (B, embed_dim)
        forget_gates = self.forget_gate(pooled_x).unsqueeze(-1)  # (B, memory_size, 1)

        # Apply forget gate with broadcasting on embed_dim
        memory = memory * (1 - forget_gates)

        # Concatenate gated input with memory along sequence dimension
        combined = torch.cat([gated_inputs, memory], dim=1)  # (B, T + memory_size, embed_dim)

        # Compress combined memory
        combined_flat = combined.view(B, -1)  # Flatten (B, (T + memory_size) * embed_dim)
        compressed = self.memory_compress(combined_flat)
        compressed = compressed.view(B, self.seq_len + self.memory_size, self.embed_dim)

        return compressed
