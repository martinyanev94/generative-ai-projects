import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cifar10_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(cifar10_dataset, batch_size=64, shuffle=True)
