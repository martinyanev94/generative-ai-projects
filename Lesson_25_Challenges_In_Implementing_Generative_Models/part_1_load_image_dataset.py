import torch
from torchvision import datasets, transforms

def load_dataset(data_dir):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    return dataset
