with torch.no_grad():
    noise = torch.randn(64, 100)  # Generate latent noise
    generated_images = generator(noise).view(-1, 1, 28, 28)

# Visualize the generated images
fig, axs = plt.subplots(8, 8, figsize=(8, 8))
for i in range(64):
    axs[i // 8, i % 8].imshow(generated_images[i].squeeze(), cmap='gray')
    axs[i // 8, i % 8].axis('off')
plt.show()
