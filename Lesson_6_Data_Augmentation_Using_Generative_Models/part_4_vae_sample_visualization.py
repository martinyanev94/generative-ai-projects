import matplotlib.pyplot as plt

# Generate new data
with torch.no_grad():
    z = torch.randn(64, 20)  # Latent space samples
    samples = vae.decode(z).view(-1, 1, 28, 28)

# Visualize the generated samples
fig, axs = plt.subplots(8, 8, figsize=(8, 8))
for i in range(64):
    axs[i // 8, i % 8].imshow(samples[i].squeeze(), cmap='gray')
    axs[i // 8, i % 8].axis('off')
plt.show()
