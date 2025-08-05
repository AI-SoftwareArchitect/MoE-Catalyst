import torch
from models.attention.directed_multihead import DirectedMultiheadAttention

def test_directed_multihead_attention():
    torch.manual_seed(0)
    batch_size = 2
    seq_len = 4
    embed_dim = 16
    num_heads = 4

    model = DirectedMultiheadAttention(embed_dim, num_heads, dropout=0.0)
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Causal mask: upper triangular mask to prevent attending future tokens
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)  # shape (1,1,T,T)
    mask = mask.expand(batch_size, num_heads, seq_len, seq_len).bool()

    output = model(x, mask=mask)

    assert output.shape == (batch_size, seq_len, embed_dim)
    print("Output shape:", output.shape)
    print("Sample output:", output[0, 0, :5])

if __name__ == "__main__":
    test_directed_multihead_attention()
