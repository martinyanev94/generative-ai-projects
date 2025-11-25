from torch.utils.data import DataLoader

# Assuming dataset is pre-defined
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

for images in data_loader:
    # Feed images to the model
