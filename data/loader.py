import torch
import re
# =============================================================================
# DATA HANDLING (Enhanced)
# =============================================================================
def load_data(file_path: str = "data.txt"):
    """Enhanced data loading with punctuation removal and better English tokenization"""
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("""Your English training data here...""")

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Remove punctuation and tokenize
    text = re.sub(r'[^\w\s]', '', text)  # Remove all punctuation
    sentences = re.split(r'[\n]', text.strip())  # Split by line for sentences
    all_words = []
    for sentence in sentences:
        words = re.findall(r"\b\w+\b", sentence.lower())
        all_words.extend(words)

    vocab = sorted(set(all_words))
    word_to_id = {word: i for i, word in enumerate(vocab)}
    id_to_word = {i: word for word, i in word_to_id.items()}

    # Convert to token sequences
    token_sequences = []
    for sentence in sentences:
        words = re.findall(r"\b\w+\b", sentence.lower())
        if words:
            tokens = [word_to_id[word] for word in words]
            token_sequences.append(tokens)

    return token_sequences, vocab, word_to_id, id_to_word


def create_batches(token_sequences, seq_len=16, batch_size=4):
    """Create training batches from token sequences"""
    batches = []

    # Create input-target pairs
    pairs = []
    for sequence in token_sequences:
        if len(sequence) > seq_len:
            for i in range(len(sequence) - seq_len):
                input_seq = sequence[i:i + seq_len]
                target_seq = sequence[i + 1:i + seq_len + 1]
                pairs.append((torch.tensor(input_seq), torch.tensor(target_seq)))

    # Group into batches
    for i in range(0, len(pairs), batch_size):
        batch_group = pairs[i:i + batch_size]
        if len(batch_group) == batch_size:  # Only use complete batches
            inputs = torch.stack([b[0] for b in batch_group])
            targets = torch.stack([b[1] for b in batch_group])
            batches.append((inputs, targets))

    return batches