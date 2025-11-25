from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten the images
])

train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
encoder = Encoder(input_dim=784, hidden_dim=400, latent_dim=20).to(device)
decoder = Decoder(latent_dim=20, hidden_dim=400, output_dim=784).to(device)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

num_epochs = 10
for epoch in range(num_epochs):
    for data, _ in train_loader:
        data = data.to(device)
        
        optimizer.zero_grad()
        mu, logvar = encoder(data)
        z = reparameterize(mu, logvar)
        recon_batch = decoder(z)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss.item()}')
