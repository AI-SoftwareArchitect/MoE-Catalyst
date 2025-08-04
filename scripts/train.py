import torch

from config.base_config import device, BrainFormerConfig
from data.loader import load_data, create_batches
from models.brainformer import BrainFormer
from sampling.text_generation import generate_text
from training.train_loop import train_model
from utils.persistence import save_model, load_model

# =============================================================================
# MAIN PROGRAM
# =============================================================================
def main():
    print("🧠 BrainFormer Advanced - Beyin Benzeri Transformer")
    print("=" * 60)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)

    while True:
        print("\n1 - Train (Gelişmiş Eğitim)")
        print("2 - Run (Gelişmiş Çalıştırma)")
        print("3 - Exit (Çıkış)")

        choice = input("\nSeçiminiz (1-3): ").strip()

        if choice == "1":
            print("\n🔄 Gelişmiş eğitim başlıyor...")

            # Load data
            token_sequences, vocab, word_to_id, id_to_word = load_data()
            print(f"📊 Veri yüklendi - Vocab: {len(vocab)}, Sequences: {len(token_sequences)}")

            # Create config with smaller sequence length
            config = BrainFormerConfig(
                vocab_size=len(vocab),
                max_seq_len=16,
                d_model=128,  # Küçültülmüş model boyutu
                num_heads=4,
                num_layers=4,
                ff_hidden_dim=512,  # Küçültülmüş FFN
                dropout=0.1
            )

            # Create model
            model = BrainFormer(config).to(device)
            param_count = sum(p.numel() for p in model.parameters())
            print(f"🏗️ Model oluşturuldu - Parametreler: {param_count:,}")

            # Create batches with smaller sequence length
            batches = create_batches(token_sequences, seq_len=16, batch_size=8)
            print(f"📦 Batch'ler hazırlandı: {len(batches)}")

            if len(batches) == 0:
                print("❌ Yeterli veri yok!")
                continue

            # Train
            train_model(model, batches, epochs=30, lr=5e-4)  # Reduced epochs

            # Save
            save_model(model, vocab, word_to_id, id_to_word)
            print("✅ Gelişmiş eğitim tamamlandı!")

        elif choice == "2":
            print("\n🚀 Gelişmiş model çalıştırılıyor...")

            # Load model
            model, vocab, word_to_id, id_to_word = load_model()

            if model is None:
                print("❌ Eğitilmiş model bulunamadı! Önce eğitim yapın.")
                continue

            print("✅ Gelişmiş model yüklendi!")
            print("🎛️ Gelişmiş sampling aktif (temperature, top-k, top-p)")

            while True:
                prompt = input("\nPrompt girin (çıkmak için 'q'): ")
                if prompt.lower() == 'q':
                    break

                # Generate with different temperatures
                print(f"\n🌡️ Temperature Karşılaştırması:")
                for temp in [0.3, 0.8, 1.2]:
                    generated_text = generate_text(model, vocab, word_to_id, id_to_word, prompt, temperature=temp)
                    print(f"T={temp}: {generated_text}")

        elif choice == "3":
            print("\n👋 Görüşmek üzere!")
            break

        else:
            print("❌ Geçersiz seçim! Lütfen 1, 2 veya 3 girin.")
