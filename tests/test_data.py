# tests/test_tokenizer.py
from data.tokenizer import Tokenizer
import os

def test_tokenizer_encode_decode():
    vocab_path = "data/vocab.json"
    if os.path.exists(vocab_path):
        os.remove(vocab_path)  # Eski vocab'ı sil

    tokenizer = Tokenizer(vocab_path)
    # Eğer Tokenizer'da force parametresi yoksa direkt çağır:
    tokenizer.build_vocab(["Merhaba dünya nasılsın iyi misin ? 🌍"])

    text = "Merhaba dünya nasılsın iyi misin ? 🌍"
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print("Tokens:", tokens)
    print("IDs:", ids)
    print("Decoded:", decoded)

    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)

if __name__ == "__main__":
    test_tokenizer_encode_decode()
