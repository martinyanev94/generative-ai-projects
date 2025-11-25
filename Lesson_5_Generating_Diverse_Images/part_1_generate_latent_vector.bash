pip install torch torchvision
import torch

# Set the random seed for reproducibility
torch.manual_seed(42)

# Generate a latent vector of size 100
latent_vector = torch.randn(1, 100)

print("Latent Vector:", latent_vector)
