import torch.optim as optim

generator = Generator(input_dim=100, output_dim=3).to(device)
discriminator = Discriminator(input_dim=3).to(device)

optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

num_epochs = 50
for epoch in range(num_epochs):
    for real_images, _ in train_loader:
        real_images = real_images.to(device)
        
        # Train Discriminator
        noise = torch.randn(real_images.size(0), 100, 1, 1).to(device)
        fake_images = generator(noise)

        real_output = discriminator(real_images)
        fake_output = discriminator(fake_images)

        d_loss = discriminator_loss(real_output, fake_output)

        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        # Train Generator
        fake_output = discriminator(fake_images)
        g_loss = generator_loss(fake_output)

        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()
