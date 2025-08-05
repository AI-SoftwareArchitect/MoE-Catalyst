from dataclasses import dataclass
import torch
# =============================================================================
# CUDA OPTIMIZATION & SETUP
# =============================================================================
# CUDA kontrolü ve optimizasyon
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cuda")
    print(f"🚀 CUDA kullanılıyor: {torch.cuda.get_device_name()}")
    # CUDA bellek optimizasyonu
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    device = torch.device("cpu")
    print("⚠️ CPU kullanılıyor")

# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class BrainFormerConfig:
    d_model: int = 16
    num_heads: int = 2
    num_layers: int = 2
    vocab_size: int = 1000
    max_seq_len: int = 16
    memory_size: int = 16
    ff_hidden_dim: int = 64
    early_exit_threshold: float = 0.9
    dropout: float = 0.1
    temperature: float = 0.8
    device: str = str(device)
