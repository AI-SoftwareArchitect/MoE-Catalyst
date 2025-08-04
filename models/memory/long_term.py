import torch
import torch.nn as nn
# =============================================================================
# ADVANCED MEMORY SYSTEMS -=LONG TERM=-
# =============================================================================
class LongTermMemory(nn.Module):
    """Enhanced LTM with attention-based retrieval"""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

        # Memory transformation layers
        self.memory_transform = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # Circular buffer for long-term storage
        self.register_buffer('ltm_buffer', torch.zeros(1024, embed_dim))
        self.register_buffer('ltm_ptr', torch.zeros(1, dtype=torch.long))

    def forward(self, x: torch.Tensor, store: bool = False) -> torch.Tensor:
        if store and self.training:
            # Store important representations
            with torch.no_grad():
                # Simple importance scoring (can be improved)
                importance = torch.norm(x, dim=-1, keepdim=True)
                important_items = x[importance.squeeze(-1) > importance.mean()]

                if len(important_items) > 0:
                    # Store in circular buffer
                    ptr = self.ltm_ptr.item()
                    batch_size = min(important_items.size(0), 32)  # Batch size limit

                    end_ptr = (ptr + batch_size) % 1024
                    if end_ptr > ptr:
                        self.ltm_buffer[ptr:end_ptr] = important_items[:batch_size]
                    else:
                        self.ltm_buffer[ptr:] = important_items[:1024 - ptr]
                        if end_ptr > 0:
                            self.ltm_buffer[:end_ptr] = important_items[1024 - ptr:batch_size]

                    self.ltm_ptr[0] = end_ptr

        return self.memory_transform(x)