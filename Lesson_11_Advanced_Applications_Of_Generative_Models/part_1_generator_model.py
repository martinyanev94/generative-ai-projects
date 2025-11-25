import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)  # Example layer

    def forward(self, x):
        x = nn.ReLU()(self.layer1(x))
        return x
