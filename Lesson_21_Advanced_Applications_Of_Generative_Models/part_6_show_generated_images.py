import matplotlib.pyplot as plt

def show_generated_images(generator, num_images=10):
    noise = torch.randn(num_images, 100, 1, 1).to(device)
    with torch.no_grad():
        fake_images = generator(noise).cpu()
    
    grid = torchvision.utils.make_grid(fake_images, nrow=5).permute(1, 2, 0)
    plt.imshow((grid + 1) / 2)  # Rescale to [0, 1]
    plt.axis('off')
    plt.show()
