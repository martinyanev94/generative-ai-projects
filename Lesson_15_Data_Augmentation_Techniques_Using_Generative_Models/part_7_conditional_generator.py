class ConditionalGenerator(nn.Module):
    def __init__(self, num_classes):
        super(ConditionalGenerator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, 100)
        self.network = nn.Sequential(
            nn.Linear(100 + 100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_embedding(labels)
        z = z + c
        return self.network(z).view(-1, 1, 28, 28)
