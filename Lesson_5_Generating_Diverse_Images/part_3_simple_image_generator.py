import torch.nn as nn

class SimpleImageGenerator(nn.Module):
    def __init__(self):
        super(SimpleImageGenerator, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 3 * 64 * 64)  # Output size for a 64x64 image with 3 channels (RGB)

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))  # Use tanh to scale output to [-1, 1]
        return x.view(-1, 3, 64, 64)  # Reshape to image dimensions

generator = SimpleImageGenerator()
generated_image = generator(latent_vector)
print("Generated Image Shape:", generated_image.shape)
