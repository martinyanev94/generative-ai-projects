class DiscriminatorWithMiniBatch(nn.Module):
    def __init__(self, batch_size):
        super(DiscriminatorWithMiniBatch, self).__init__()
        self.batch_size = batch_size
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28 + batch_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, context):
        combined = torch.cat((img.view(self.batch_size, -1), context), dim=1)
        return self.model(combined)
