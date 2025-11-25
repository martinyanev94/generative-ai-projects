from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = DataLoader(mnist_dataset, batch_size=128, shuffle=True)

# Initialize the VAE and optimizer
vae = VAE(input_dim=784, latent_dim=20)
optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    for batch_idx, (data, _) in enumerate(data_loader):
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data)
        loss = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
