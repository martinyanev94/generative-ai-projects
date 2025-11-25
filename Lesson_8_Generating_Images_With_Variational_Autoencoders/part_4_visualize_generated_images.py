import matplotlib.pyplot as plt

with torch.no_grad():
    # Sample random points in the latent space
    z_sample = torch.randn(64, 20).to(device)
    generated_images = decoder(z_sample).cpu()

    # Visualize the generated images
    grid_img = torchvision.utils.make_grid(generated_images.view(-1, 1, 28, 28), nrow=8, padding=2)
    plt.imshow(grid_img.numpy().transpose((1, 2, 0)), cmap='gray')
    plt.axis('off')
    plt.show()
