disc = Discriminator()
gen = Generator()
d_optimizer = torch.optim.Adam(disc.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(gen.parameters(), lr=0.0002)

for epoch in range(1, 11):
    for batch_idx, (data, _) in enumerate(train_loader):
        # Train Discriminator
        d_optimizer.zero_grad()
        real_images = data
        real_labels = torch.ones(data.size(0), 1)
        fake_images = gen(torch.randn(data.size(0), 100))
        fake_labels = torch.zeros(data.size(0), 1)

        real_loss = nn.BCELoss()(disc(real_images), real_labels)
        fake_loss = nn.BCELoss()(disc(fake_images.detach()), fake_labels)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        g_optimizer.zero_grad()
        g_loss = nn.BCELoss()(disc(fake_images), real_labels)
        g_loss.backward()
        g_optimizer.step()

    print(f'Epoch {epoch}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')
