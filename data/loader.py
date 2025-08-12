import torch
from data.text_dataset import TextDataset
from data.tokenizer import Tokenizer  # Yeni sınıfın burada

# =============================================================================
# DATA HANDLING (PRODUCTION READY)
# =============================================================================
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
