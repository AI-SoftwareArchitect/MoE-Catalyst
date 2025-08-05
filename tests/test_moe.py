import torch
from models.moe.moe_layer import MoELayer

def test_moe():
    torch.manual_seed(0)
    batch_size = 8
    embed_dim = 16
    num_experts = 4
    expert_hidden_dim = 32
    top_k = 2

    model = MoELayer(embed_dim, num_experts, expert_hidden_dim, top_k=top_k, dropout=0.0)
    x = torch.randn(batch_size, embed_dim)

    output = model(x)

    assert output.shape == (batch_size, embed_dim), f"Expected output shape {(batch_size, embed_dim)}, got {output.shape}"
    print("MoE output shape:", output.shape)
    print("Sample output:", output[0, :5])

if __name__ == "__main__":
    test_moe()
