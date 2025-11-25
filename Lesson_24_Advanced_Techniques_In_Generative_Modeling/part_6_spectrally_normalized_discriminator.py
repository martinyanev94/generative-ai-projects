class SpectrallyNormalizedDiscriminator(nn.Module):
    def __init__(self):
        super(SpectrallyNormalizedDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )
        self.apply_spectral_norm()

    def apply_spectral_norm(self):
        for layer in self.model:
            layer = nn.utils.spectral_norm(layer)

    def forward(self, img):
        return self.model(img.view(img.size(0), -1))
