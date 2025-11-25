import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size must be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        N, seq_length, _ = x.shape
        value_len, key_len, query_len = seq_length, seq_length, seq_length

        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        values = values.view(N, value_len, self.heads, self.head_dim)
        keys = keys.view(N, key_len, self.heads, self.head_dim)
        queries = queries.view(N, query_len, self.heads, self.head_dim)

        values = values.permute(0, 2, 1, 3)  # (N, heads, value_len, head_dim)
        keys = keys.view(N, key_len, self.heads, self.head_dim)
        queries = queries.permute(0, 2, 1, 3) # (N, heads, query_len, head_dim)

        energy = torch.einsum("nqhd,nkhd->nqkh", [queries, keys])  # (N, query_len, key_len, heads)
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nqkh,nvhd->nqhd", [attention, values]).view(N, seq_length, self.heads * self.head_dim)
        return self.fc_out(out)
