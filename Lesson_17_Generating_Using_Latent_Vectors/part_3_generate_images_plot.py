import matplotlib.pyplot as plt

def generate_images(model, num_images=10):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_images, 20)  # Sample from a normal distribution
        generated_images = model.decoder(z).view(-1, 1, 28, 28)
        return generated_images

images = generate_images(model)
for i in range(len(images)):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i].numpy().squeeze(), cmap='gray')
    plt.axis('off')
plt.show()
