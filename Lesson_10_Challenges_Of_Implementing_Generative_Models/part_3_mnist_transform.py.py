from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist = datasets.MNIST(root='data', train=True, transform=transform, download=True)
dataloader = DataLoader(mnist, batch_size=64, shuffle=True)
