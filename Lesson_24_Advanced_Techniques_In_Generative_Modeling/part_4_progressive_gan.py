class ProgressiveGenerator(nn.Module):
    def __init__(self):
        super(ProgressiveGenerator, self).__init__()
        self.initial = nn.Linear(z_dim, 128)
        self.model_4x4 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28)
        )

    def forward(self, z):
        x = self.initial(z)
        return self.model_4x4(x).reshape(-1, 1, 4, 4)

class ProgressiveDiscriminator(nn.Module):
    def __init__(self):
        super(ProgressiveDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 4, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img.view(img.size(0), -1))
