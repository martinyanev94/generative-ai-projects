def visualize_reconstructed_images(vae, num_images=10):
    with torch.no_grad():
        z = torch.randn(num_images, 20)
        generated_images = vae.decode(z).view(-1, 1, 28, 28)
        grid = make_grid(generated_images, nrow=5, normalize=True)
        plt.imshow(grid.permute(1, 2, 0))
        plt.title('Generated Images from VAE')
        plt.axis('off')
        plt.show()
