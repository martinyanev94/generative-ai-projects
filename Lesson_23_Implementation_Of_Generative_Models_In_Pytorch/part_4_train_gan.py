for epoch in range(epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        batch_size = real_imgs.shape[0]
        
        # Create labels
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        # Train Discriminator
        optimizer_D.zero_grad()
        outputs = discriminator(real_imgs)
        d_loss_real = criterion(outputs, real_labels)
        
        z = torch.randn(batch_size, z_dim)
        fake_imgs = generator(z)
        outputs = discriminator(fake_imgs.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        outputs = discriminator(fake_imgs)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch}/{epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}")
