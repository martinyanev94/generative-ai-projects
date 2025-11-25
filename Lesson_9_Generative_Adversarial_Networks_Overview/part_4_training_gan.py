transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist = datasets.MNIST(root='data', train=True, transform=transform, download=True)
dataloader = DataLoader(mnist, shuffle=True, batch_size=64)
generator = Generator()
discriminator = Discriminator()

criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

for epoch in range(50):
    for i, (imgs, _) in enumerate(dataloader):
        batch_size = imgs.shape[0]

        # Labels for real and fake images
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Train Discriminator
        optimizer_D.zero_grad()
        outputs = discriminator(imgs)
        d_loss_real = criterion(outputs, real_labels)

        z = torch.randn(batch_size, 100)
        fake_images = generator(z)
        outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()

    print(f'Epoch [{epoch + 1}/50], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')
def visualize(generator, epoch):
    z = torch.randn(64, 100)
    fake_images = generator(z)
    fake_images = fake_images.view(-1, 28, 28).detach().numpy()
    plt.figure(figsize=(10, 10))
    for i in range(64):
        plt.subplot(8, 8, i + 1)
        plt.imshow(fake_images[i], cmap='gray')
        plt.axis('off')
    plt.suptitle(f'Epoch: {epoch}')
    plt.show()
