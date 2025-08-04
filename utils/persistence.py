import os
import torch
from config.base_config import BrainFormerConfig, device
from models.brainformer import BrainFormer
# =============================================================================
# MODEL PERSISTENCE
# =============================================================================
def save_model(model, vocab, word_to_id, id_to_word, path="brainformer_advanced.pt"):
    """Save model with enhanced metadata"""
    # config'i dict olarak kaydet
    config_dict = model.config.__dict__ if hasattr(model.config, '__dict__') else dict(model.config)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'word_to_id': word_to_id,
        'id_to_word': id_to_word,
        'config': config_dict,
        'model_class': 'BrainFormer'
    }, path)
    print(f"✅ Gelişmiş model kaydedildi: {path}")

def load_model(path="brainformer_advanced.pt"):
    """Load model with error handling"""
    if not os.path.exists(path):
        return None, None, None, None

    try:
        # PyTorch 2.6+ için weights_only=False ekle
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config_dict = checkpoint['config']
        # Dict'ten BrainFormerConfig nesnesi oluştur
        config = BrainFormerConfig(**config_dict)
        config.device = str(device)

        model = BrainFormer(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        return model, checkpoint['vocab'], checkpoint['word_to_id'], checkpoint['id_to_word']
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        return None, None, None, None
