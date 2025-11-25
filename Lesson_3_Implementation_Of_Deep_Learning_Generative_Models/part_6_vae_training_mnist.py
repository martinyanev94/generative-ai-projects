# Load the MNIST dataset
data_loader = torch.utils.data.DataLoader(mnist, batch_size=64, shuffle=True)

# Initialize the VAE and set up the optimizer
vae = VAE()
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    for real_images, _ in data_loader:
        real_images = real_images.view(-1, 784)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        reconstructed, mu, logvar = vae(real_images)

        # Compute the reconstruction loss
        reconstruction_loss = nn.functional.binary_cross_entropy(reconstructed, real_images, reduction='sum')
        
        # Compute the KL divergence loss
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        loss = reconstruction_loss + kl_divergence
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
