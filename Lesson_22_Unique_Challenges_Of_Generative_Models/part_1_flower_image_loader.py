from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a transformation to normalize the data and resize images.
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load your dataset with these transformations.
train_dataset = datasets.ImageFolder(root='path_to_flower_images', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
