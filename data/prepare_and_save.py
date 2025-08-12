import re
import numpy as np

from data.tokenizer import Tokenizer


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?]', '', text)
    return text.strip()

def prepare_and_save_data(file_path="data.txt", token_output="tokens.npy", vocab_output="vocab.json"):
    tokenizer = Tokenizer(vocab_output)

    print("Veri yükleniyor...")
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()

    print("Veri temizleniyor...")
    lines = [clean_text(line) for line in raw_lines]

    print("Vocab oluşturuluyor...")
    tokenizer.build_vocab(lines)

    all_tokens = []
    for line in lines:
        all_tokens.extend(tokenizer.encode(line))

    token_array = np.array(all_tokens, dtype=np.int32)
    np.save(token_output, token_array)
    print("Token'lar ve vocab kaydedildi.")
