# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

def train(generator, dataloader, epochs=50):
    for epoch in range(epochs):
        for real_images, _ in dataloader:
            batch_size = real_images.size(0)

            # Generate random latent vectors
            latent_vectors = torch.randn(batch_size, 100)

            # Generate fake images
            fake_images = generator(latent_vectors)

            # Calculate loss (simplified, assuming we have a discriminator)
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # Loss for real images
            real_loss = criterion(discriminator(real_images), real_labels)
            # Loss for fake images
            fake_loss = criterion(discriminator(fake_images.detach()), fake_labels)

            # Backpropagation
            optimizer.zero_grad()
            total_loss = real_loss + fake_loss
            total_loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch}/{epochs}], Loss: {total_loss.item():.4f}")

train(generator, dataloader)
