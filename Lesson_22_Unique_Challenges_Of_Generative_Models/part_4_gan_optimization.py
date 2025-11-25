import torch.optim as optim

generator = Generator(input_dim=100, output_dim=3)
discriminator = Discriminator(input_dim=3)

# Using different learning rates for generator and discriminator
optimizer_g = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
