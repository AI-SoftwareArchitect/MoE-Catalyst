import torch
import torch.nn.functional as F

# =============================================================================
# ADVANCED SAMPLING
# =============================================================================
def sample_with_temperature(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 50,
                            top_p: float = 0.9) -> torch.Tensor:
    """Advanced sampling with temperature, top-k, and top-p (nucleus sampling)"""
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # Apply temperature
    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        top_k_logits, top_k_indices = torch.topk(logits, top_k)
        logits_filtered = torch.full_like(logits, -float('inf'))
        logits_filtered.scatter_(-1, top_k_indices, top_k_logits)
        logits = logits_filtered

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        # İşlemi her örnek için ayrı yap
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Top-p maskesi oluştur
        sorted_indices_to_remove = cumulative_probs > top_p
        # Her satırda ilk token daima tutulmalı
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Maskeyi orijinal indekslere uygula
        for batch_idx in range(logits.size(0)):
            indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
            logits[batch_idx, indices_to_remove] = -float('inf')

    # Sample from the filtered distribution
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
