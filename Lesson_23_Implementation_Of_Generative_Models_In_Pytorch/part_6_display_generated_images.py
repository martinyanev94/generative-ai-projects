def show_generated_images(generator, num_images=25):
    z = torch.randn(num_images, z_dim)
    with torch.no_grad():
        generated_images = generator(z).cpu()
    
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(5, 5, i + 1)
        plt.imshow(generated_images[i].squeeze(0), cmap='gray')
        plt.axis('off')
    plt.show()

show_generated_images(generator)
