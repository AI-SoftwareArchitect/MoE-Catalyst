import re
import os
from data import emoji_pattern
class Tokenizer:
    def __init__(self, vocab_path="vocab.json"):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_path = vocab_path
        if os.path.exists(vocab_path):
            self.load_vocab()

    def build_vocab(self, text_lines):
        unique_words = set()
        for line in text_lines:
            unique_words.update(self.tokenize(line))

        special_tokens = ["[PAD]", "[UNK]"]
        vocab = special_tokens + sorted(unique_words)
        self.word_to_id = {word: idx for idx, word in enumerate(vocab)}
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}

        dir_path = os.path.dirname(self.vocab_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        self.save_vocab()

    def save_vocab(self):
        import json
        with open(self.vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.word_to_id, f, ensure_ascii=False, indent=2)

    def load_vocab(self):
        import json
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            self.word_to_id = json.load(f)
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}

    def tokenize(self, text):
        text = text.lower().strip()
        text = emoji_pattern.EMOJI_PATTERN.sub(lambda match: f" {match.group(0)} ", text)
        words = re.findall(r"\b\w+\b|[\S]", text)
        return [w.strip() for w in words if w.strip()]

    def encode(self, text):
        return [self.word_to_id.get(w, self.word_to_id["[UNK]"]) for w in self.tokenize(text)]

    def decode(self, ids):
        return ' '.join(self.id_to_word.get(i, "[UNK]") for i in ids)
