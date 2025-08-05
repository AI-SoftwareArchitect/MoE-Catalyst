import torch
import torch.nn as nn
import torch.nn.functional as F
# =============================================================================
# ADVANCED MEMORY SYSTEMS -=LONG TERM=-
# =============================================================================
class LongTermMemory(nn.Module):
    """Enhanced Long Term Memory with learnable importance scoring and attention-based retrieval"""

    def __init__(self, embed_dim: int, buffer_size: int = 1024, device=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.buffer_size = buffer_size
        self.device = device or torch.device("cpu")

        # Learnable importance scorer (sigmoid gating)
        self.importance_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

        # Memory transformation (to get query/key/value vectors)
        self.memory_transform = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # Circular buffer and pointer for storage
        self.register_buffer('ltm_buffer', torch.zeros(buffer_size, embed_dim))
        self.register_buffer('ltm_ptr', torch.zeros(1, dtype=torch.long))

        # Decay factor to forget old memories gradually
        self.decay_rate = 0.99

    def store(self, x: torch.Tensor):
        """Store important representations into circular buffer"""
        with torch.no_grad():
            importance_scores = self.importance_gate(x).squeeze(-1)  # (B*T)
            threshold = importance_scores.mean()
            important_mask = importance_scores > threshold
            important_items = x[important_mask]

            if important_items.size(0) == 0:
                return  # Nothing important to store

            ptr = self.ltm_ptr.item()
            batch_size = important_items.size(0)

            end_ptr = (ptr + batch_size) % self.buffer_size

            if end_ptr > ptr:
                self.ltm_buffer[ptr:end_ptr] = important_items[:batch_size].to(self.device)
            else:
                first_len = self.buffer_size - ptr
                self.ltm_buffer[ptr:] = important_items[:first_len].to(self.device)
                self.ltm_buffer[:end_ptr] = important_items[first_len:batch_size].to(self.device)

            self.ltm_ptr[0] = end_ptr

    def retrieve(self, x: torch.Tensor, top_k: int = 16):
        """Retrieve relevant memories for input x using cosine similarity"""
        if self.ltm_ptr.item() == 0:
            return torch.zeros_like(x)

        # Normalize queries and memory
        queries = F.normalize(x, dim=-1)  # (B, T, C)
        memories = F.normalize(self.ltm_buffer, dim=-1)  # (buffer_size, C)

        # Compute cosine similarity (B, T, buffer_size)
        scores = torch.einsum('btc,mc->btm', queries, memories)

        # Top-k retrieval
        topk_scores, topk_indices = torch.topk(scores, k=min(top_k, self.ltm_ptr.item()), dim=-1)

        # Weighted sum of retrieved memories
        retrieved = torch.zeros_like(x)
        for b in range(x.size(0)):
            for t in range(x.size(1)):
                idxs = topk_indices[b, t]
                wts = F.softmax(topk_scores[b, t], dim=-1)
                mems = self.ltm_buffer[idxs]
                retrieved[b, t] = (mems * wts.unsqueeze(-1)).sum(dim=0)

        return retrieved

    def forward(self, x: torch.Tensor, store: bool = False, retrieve: bool = False):
        """
        x: input tensor (B, T, C)
        store: whether to store important features into memory
        retrieve: whether to retrieve relevant memories and add to input
        """
        if store and self.training:
            self.store(x.view(-1, self.embed_dim))

            # Apply decay to buffer to slowly forget old memories
            self.ltm_buffer.mul_(self.decay_rate)

        mem_output = self.memory_transform(x)

        if retrieve:
            retrieved_mem = self.retrieve(x)
            mem_output = mem_output + retrieved_mem

        return mem_output