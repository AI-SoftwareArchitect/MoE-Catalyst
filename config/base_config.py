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
    vocab_size: int = 15000
    d_model: int = 128
    max_seq_len: int = 512
    num_heads: int = 4
    num_layers: int = 4
    ff_hidden_dim: int = 512
    dropout: float = 0.1

    # Bellek/erken çıkış
    memory_size: int = 8
    early_exit_threshold: float = 0.95
    ltm_buffer_size: int = 512
    relative_max_pos: int = 256

    # MoE ayarları
    use_moe: bool = True                 # MoE katmanını aç/kapat
    moe_num_experts: int = 4             # ≥ top_k olmalı (öneri: 4 veya 8)
    moe_expert_hidden: int = 512         # tipik olarak 2–4× d_model
    moe_top_k: int = 2                   # istenen top-k
    moe_expert_dropout: float = 0.1      # uzman çıkışlarına dropout
    moe_lambda: float = 0.01             # aux (load-balance) loss katsayısı

    def __post_init__(self):
        assert 1 <= self.moe_top_k <= self.moe_num_experts, \
            f"moe_top_k ({self.moe_top_k}) moe_num_experts ({self.moe_num_experts}) değerini aşamaz"