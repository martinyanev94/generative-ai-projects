import matplotlib.pyplot as plt

def plot_generated_images(generator, num_images=25):
    z = torch.randn(num_images, 100)  # Generate random noise
    generated_images = generator(z).detach().numpy()

    plt.figure(figsize=(5, 5))
    for i in range(num_images):
        plt.subplot(5, 5, i + 1)
        plt.imshow(generated_images[i][0], cmap='gray')
        plt.axis('off')
    plt.show()

plot_generated_images(generator)
