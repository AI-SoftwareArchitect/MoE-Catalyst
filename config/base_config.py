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
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    vocab_size: int = 1000
    max_seq_len: int = 512
    memory_size: int = 128
    ff_hidden_dim: int = 1024  # FFN genişliği
    early_exit_threshold: float = 0.9
    dropout: float = 0.3
    temperature: float = 0.8
    device: str = str(device)
