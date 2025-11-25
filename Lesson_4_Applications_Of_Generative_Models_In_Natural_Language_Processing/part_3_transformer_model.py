class Transformer(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, vocab_size, max_length):
        super(Transformer, self).__init__()
        self.encoder = nn.Embedding(vocab_size, embed_size)
        self.transformer_blocks = nn.ModuleList(
            [SelfAttention(embed_size, num_heads) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.encoder(x)

        for transformer in self.transformer_blocks:
            x = transformer(x)

        return self.fc_out(x)
