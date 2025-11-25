from torchvision import transforms

data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

augmented_dataset = CustomDataset(transform=data_transforms)
data_loader = DataLoader(augmented_dataset, batch_size=64, shuffle=True)
