import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(784, 256)  # Input is a 784-dimensional vector (28x28 images)
        self.fc21 = nn.Linear(256, 20)   # Mean of the latent space
        self.fc22 = nn.Linear(256, 20)   # Log variance of the latent space

    def forward(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(20, 256)     # Starting point to expand from the latent representation
        self.fc4 = nn.Linear(256, 784)     # Final output to reconstruct the image

    def forward(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
