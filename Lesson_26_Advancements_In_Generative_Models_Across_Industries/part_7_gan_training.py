generator = Generator()
discriminator = Discriminator()

criterion = nn.BCELoss()
optimizer_gen = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

for epoch in range(100):
    for _ in range(imgs_per_batch):
        # Train Discriminator
        optimizer_disc.zero_grad()
        real_imgs = get_real_imgs(...)  # Load a batch of real images
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        output_real = discriminator(real_imgs)
        loss_real = criterion(output_real, real_labels)

        z = torch.randn(batch_size, 100)
        fake_imgs = generator(z)
        output_fake = discriminator(fake_imgs.detach())
        loss_fake = criterion(output_fake, fake_labels)

        d_loss = loss_real + loss_fake
        d_loss.backward()
        optimizer_disc.step()

        # Train Generator
        optimizer_gen.zero_grad()
        output_fake = discriminator(fake_imgs)
        g_loss = criterion(output_fake, real_labels)  # We want to fool the discriminator
        g_loss.backward()
        optimizer_gen.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")
