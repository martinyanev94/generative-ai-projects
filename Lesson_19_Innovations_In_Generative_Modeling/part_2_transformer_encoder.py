import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, n_heads, hidden_dim, output_dim):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.self_attention = nn.MultiheadAttention(emb_dim, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        x = self.embedding(x)
        attn_output, _ = self.self_attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        x = self.feed_forward(x)
        return self.layer_norm2(x)
