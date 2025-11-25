import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 128, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, output_dim, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.Dropout(p=0.3)  # Adding dropout to prevent overfitting
        )

    def forward(self, x):
        return self.model(x)
