class FashionGenerator(nn.Module):
    def __init__(self):
        super(FashionGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, 128, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=7, stride=2, padding=3),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)
