import torch.optim as optim

generator = ...  # Your generator model
discriminator = ...  # Your discriminator model

optim_gen = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optim_disc = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
