device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move models to GPU
generator.to(device)
discriminator.to(device)
vae.to(device)

# Example data on GPU
random_noise = torch.randn((64, 100)).to(device)
