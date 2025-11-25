class NoisyGenerator(nn.Module):
    def __init__(self):
        super(NoisyGenerator, self).__init__()
        self.fc = nn.Linear(100, 256)

    def forward(self, z):
        z_noise = z + torch.randn_like(z) * 0.1  # Adding Gaussian noise
        return torch.relu(self.fc(z_noise))
