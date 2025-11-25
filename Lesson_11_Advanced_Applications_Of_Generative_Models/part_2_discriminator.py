class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)  # Example layer

    def forward(self, x):
        x = nn.LeakyReLU(0.2)(self.layer1(x))
        return x
