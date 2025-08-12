import re
import os
import json
try:
    import tiktoken
except ImportError:
    tiktoken = None

from data import emoji_pattern

class Tokenizer:
    def __init__(self, vocab_path="vocab.json", model_name=None):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_path = vocab_path
        self.use_tiktoken = False
        self.tiktoken_encoder = None

        if model_name and tiktoken:
            try:
                self.tiktoken_encoder = tiktoken.get_encoding(model_name)
                self.use_tiktoken = True
                print(f"[Tokenizer] Tiktoken encoding yüklendi: {model_name}")
            except Exception as e:
                print(f"[Tokenizer] Tiktoken model yüklenemedi: {e}, eski mod kullanılacak.")

        if not self.use_tiktoken and os.path.exists(vocab_path):
            self.load_vocab()

    def build_vocab(self, text_lines, max_vocab_size=15000):
        if self.use_tiktoken:
            print("[Tokenizer] build_vocab tiktoken modunda kullanılmaz.")
            return

        unique_words = set()
        for line in text_lines:
            unique_words.update(self.tokenize(line))

        special_tokens = ["[PAD]", "[UNK]"]
        limited_words = sorted(unique_words)[: max_vocab_size - len(special_tokens)]
        vocab = special_tokens + limited_words

        self.word_to_id = {word: idx for idx, word in enumerate(vocab)}
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}

        dir_path = os.path.dirname(self.vocab_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        self.save_vocab()

    def save_vocab(self):
        if self.use_tiktoken:
            print("[Tokenizer] save_vocab tiktoken modunda çalışmaz.")
            return
        with open(self.vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.word_to_id, f, ensure_ascii=False, indent=2)

    def load_vocab(self):
        if self.use_tiktoken:
            print("[Tokenizer] load_vocab tiktoken modunda çalışmaz.")
            return
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            self.word_to_id = json.load(f)
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}

    def tokenize(self, text):
        if self.use_tiktoken:
            ids = self.tiktoken_encoder.encode(text, allowed_special="all")
            return [self.tiktoken_encoder.decode([i]) for i in ids]
        else:
            text = text.lower().strip()
            text = emoji_pattern.EMOJI_PATTERN.sub(lambda match: f" {match.group(0)} ", text)
            words = re.findall(r"\b\w+\b|[\S]", text)
            return [w.strip() for w in words if w.strip()]

    def encode(self, text):
        if self.use_tiktoken:
            try:
                return self.tiktoken_encoder.encode(text, allowed_special="all")
            except KeyError:
                # Karakter-level fallback
                print("[Tokenizer] Bilinmeyen token bulundu, karakter-level fallback kullanılıyor.")
                return [self.tiktoken_encoder.encode(ch, allowed_special="all")[0] for ch in text]
        return [self.word_to_id.get(w, self.word_to_id["[UNK]"]) for w in self.tokenize(text)]

    def decode(self, ids):
        if self.use_tiktoken:
            return self.tiktoken_encoder.decode(ids)
        return ' '.join(self.id_to_word.get(i, "[UNK]") for i in ids)
