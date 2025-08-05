import re
import torch
from config.base_config import device
from sampling.sampler import sample_with_temperature
# =============================================================================
# ADVANCED INFERENCE
# =============================================================================
def generate_text(model, vocab, word_to_id, id_to_word, prompt="hello", max_length=30, temperature=0.8, max_seq_len=16,
                  pad_token_id=0):
    model.eval()
    words = re.findall(r"\b\w+\b", prompt.lower())
    input_ids = [word_to_id[word] for word in words if word in word_to_id]
    if not input_ids:
        return "<No valid prompt tokens in vocab!>"
    generated = input_ids.copy()
    with torch.no_grad():
        for _ in range(max_length):
            # Slice last max_seq_len tokens
            tokens = generated[-max_seq_len:]
            # Padding if tokens less than max_seq_len
            if len(tokens) < max_seq_len:
                tokens = [pad_token_id] * (max_seq_len - len(tokens)) + tokens

            input_tensor = torch.tensor(tokens).unsqueeze(0).to(device)  # (1, max_seq_len)

            logits = model(input_tensor)  # [1, seq_len, vocab_size]
            next_token_logits = logits[0, -1, :]
            next_token_id = sample_with_temperature(
                next_token_logits.unsqueeze(0),
                temperature=temperature,
                top_k=50,
                top_p=0.9
            ).item()
            generated.append(next_token_id)
            if len(generated) >= 50:
                break
    generated_words = [id_to_word.get(token_id, "<UNK>") for token_id in generated]
    return " ".join(generated_words)
