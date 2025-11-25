class ConditionalGenerator(nn.Module):
    def __init__(self):
        super(ConditionalGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(100 + 10, 256),  # Assuming one-hot encoding for the labels
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28), 
            nn.Tanh()
        )

    def forward(self, z, labels):
        z = torch.cat((z, labels), dim=1)
        return self.fc(z).view(-1, 1, 28, 28)

class ConditionalDiscriminator(nn.Module):
    def __init__(self):
        super(ConditionalDiscriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28 + 10, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img = img.view(-1, 28 * 28)
        combined = torch.cat((img, labels), dim=1)
        return self.fc(combined)
