pip install torch torchvision
import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc21 = nn.Linear(256, 20)  # Mean
        self.fc22 = nn.Linear(256, 20)  # Log Variance

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc21(x), self.fc22(x)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(20, 256)
        self.fc4 = nn.Linear(256, 28 * 28)

    def forward(self, z):
        z = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(z))

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x.view(-1, 28 * 28))
        z = self.reparameterize(mu, logvar)
        return self.decoder(z.view(-1, 20)), mu, logvar

model = VAE()
