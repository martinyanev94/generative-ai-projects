from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FakeData(transform=transform)  # Replace with your actual dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
