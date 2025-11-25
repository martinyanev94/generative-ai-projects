import torch.optim as optim

def train(generator, discriminator, data_loader, num_epochs):
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for real_images, _ in data_loader:
            # Train the discriminator
            optimizer_d.zero_grad()
            noise = torch.randn(real_images.size(0), 3, 256, 256)  # Example noise
            fake_images = generator(noise)
            real_loss = criterion(discriminator(real_images), torch.ones_like(real_images))
            fake_loss = criterion(discriminator(fake_images.detach()), torch.zeros_like(fake_images))
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            # Train the generator
            optimizer_g.zero_grad()
            g_loss = criterion(discriminator(fake_images), torch.ones_like(fake_images))
            g_loss.backward()
            optimizer_g.step()
