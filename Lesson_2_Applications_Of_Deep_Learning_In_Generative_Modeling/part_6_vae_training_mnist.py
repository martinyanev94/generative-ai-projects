from torchvision import datasets, transforms
from torch.optim import Adam

batch_size = 64
learning_rate = 0.001
num_epochs = 20

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = VAE()
optimizer = Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    for data in train_loader:
        img, _ = data

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(img.view(-1, 784))
        loss = loss_function(recon_batch, img.view(-1, 784), mu, logvar)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
