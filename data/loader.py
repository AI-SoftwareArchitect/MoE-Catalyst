import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data.text_dataset import TextDataset
from data.tokenizer import Tokenizer  # Yeni sınıfın burada
from data.emoji_pattern import EMOJI_PATTERN  # Gerekirse, ama tokenizer sınıfına gömdüysen burada gerekmez

# =============================================================================
# DATA HANDLING (PRODUCTION READY)
# =============================================================================

def prepare_and_save_data(file_path="data.txt", token_output="tokens.npy", vocab_output="vocab.json"):
    tokenizer = Tokenizer(vocab_output)

    print("Veri yükleniyor...")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print("Vocab oluşturuluyor...")
    tokenizer.build_vocab(lines)

    all_tokens = []
    for line in lines:
        all_tokens.extend(tokenizer.encode(line))

    token_array = np.array(all_tokens, dtype=np.int32)
    np.save(token_output, token_array)
    print("Token'lar ve vocab kaydedildi.")


def load_data(token_file_path="tokens.npy", vocab_path="vocab.json", seq_len=16):
    tokenizer = Tokenizer(vocab_path)
    dataset = TextDataset(token_file_path, seq_len)
    vocab = list(tokenizer.word_to_id.keys())
    word_to_id = tokenizer.word_to_id
    id_to_word = tokenizer.id_to_word
    return dataset, vocab, word_to_id, id_to_word

def create_batches(dataset, batch_size, shuffle=True):
    generator = torch.Generator(device='cuda')  # CPU'da olmalı
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,          # Multiprocessing sorunları için 0 yap
        pin_memory=True,
        generator=generator     # CPU cihazında generator
    )
