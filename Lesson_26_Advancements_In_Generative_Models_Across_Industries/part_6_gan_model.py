import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Dense(256, input_size=100),
            nn.ReLU(),
            nn.Dense(512),
            nn.ReLU(),
            nn.Dense(1024),
            nn.ReLU(),
            nn.Dense(1, activation='tanh')  # Assuming a 1D image
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Dense(512, input_size=1),
            nn.LeakyReLU(0.2),
            nn.Dense(256),
            nn.LeakyReLU(0.2),
            nn.Dense(1, activation='sigmoid')
        )

    def forward(self, img):
        return self.model(img)
