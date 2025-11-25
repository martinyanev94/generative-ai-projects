def train_gan(generator, discriminator, epochs=100, batch_size=64):
    for epoch in range(epochs):
        for _ in range(batch_size):
            # Generate random noise
            z = torch.randn(batch_size, 100).to(device)
            fake_images = generator(z)

            # Create labels for real and fake images
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            discriminator.zero_grad()
            real_output = discriminator(real_images)  # Assume real_images is defined
            d_loss_real = criterion(real_output, real_labels)
            fake_output = discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            generator.zero_grad()
            output = discriminator(fake_images)
            g_loss = criterion(output, real_labels)
            g_loss.backward()
            g_optimizer.step()
