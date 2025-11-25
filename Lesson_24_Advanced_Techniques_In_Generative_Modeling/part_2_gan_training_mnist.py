train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def get_labels(batch_size):
    return torch.eye(10)[torch.randint(0, 10, (batch_size,))]
for epoch in range(epochs):
    for i, (real_imgs, _) in enumerate(train_loader):
        batch_size = real_imgs.size(0)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Get one-hot labels
        labels = get_labels(batch_size)

        optimizer_D.zero_grad()
        outputs = conditional_discriminator(real_imgs, labels)
        d_loss_real = criterion(outputs, real_labels)

        z = torch.randn(batch_size, z_dim)
        fake_imgs = conditional_generator(z, labels)
        outputs = conditional_discriminator(fake_imgs.detach(), labels)
        d_loss_fake = criterion(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        outputs = conditional_discriminator(fake_imgs, labels)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch}/{epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}")
