class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3*64*64, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 3*64*64),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x.view(-1, 3*64*64))
        decoded = self.decoder(encoded)
        return decoded.view(-1, 3, 64, 64)

autoencoder = AutoEncoder().to(device)
