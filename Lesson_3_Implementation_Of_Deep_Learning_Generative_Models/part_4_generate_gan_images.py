import matplotlib.pyplot as plt

def generate_and_save_images(generator, epoch, num_images=10):
    z = torch.randn(num_images, 100)
    generated_images = generator(z).view(-1, 1, 28, 28).detach()
    grid = make_grid(generated_images, nrow=5, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(f'Epoch {epoch}')
    plt.axis('off')
    plt.savefig(f'gan_image_epoch_{epoch}.png')
    plt.show()
if (epoch+1) % 10 == 0:
    generate_and_save_images(generator, epoch+1)
