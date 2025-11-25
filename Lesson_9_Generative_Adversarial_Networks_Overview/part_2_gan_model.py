import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
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
            nn.Dense(784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Dense(512, input_size=784),
            nn.LeakyReLU(0.2),
            nn.Dense(256),
            nn.LeakyReLU(0.2),
            nn.Dense(1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)
