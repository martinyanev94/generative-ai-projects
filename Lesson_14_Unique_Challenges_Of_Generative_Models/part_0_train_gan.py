import torch

def train_gan(generator, discriminator, data_loader, epochs=100):
    for epoch in range(epochs):
        for real_images in data_loader:
            # Generate noise for the generator
            noise = torch.randn(real_images.size(0), 100)
            fake_images = generator(noise)

            # Train the discriminator
            discriminator.zero_grad()
            real_labels = torch.ones(real_images.size(0), 1)
            fake_labels = torch.zeros(real_images.size(0), 1)

            real_output = discriminator(real_images)
            fake_output = discriminator(fake_images.detach())
            
            d_loss_real = criterion(real_output, real_labels)
            d_loss_fake = criterion(fake_output, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # Train the generator
            generator.zero_grad()
            output = discriminator(fake_images)
            g_loss = criterion(output, real_labels)
            g_loss.backward()
            g_optimizer.step()
