criterion = nn.BCELoss()
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
for epoch in range(30):
    for batch_idx, (real_images, _) in enumerate(data_loader):
        batch_size = real_images.size(0)

        # Train Discriminator
        discriminator_optimizer.zero_grad()
        
        # Training with real images
        real_labels = torch.ones(batch_size, 1)
        real_outputs = discriminator(real_images.view(batch_size, -1))
        real_loss = criterion(real_outputs, real_labels)

        # Training with fake images
        noise = torch.randn(batch_size, 100)
        fake_images = generator(noise)
        fake_labels = torch.zeros(batch_size, 1)
        fake_outputs = discriminator(fake_images.detach())
        fake_loss = criterion(fake_outputs, fake_labels)

        # Backpropagation
        d_loss = real_loss + fake_loss
        d_loss.backward()
        discriminator_optimizer.step()

        # Train Generator
        generator_optimizer.zero_grad()
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        generator_optimizer.step()

    print(f'Epoch {epoch + 1}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')
