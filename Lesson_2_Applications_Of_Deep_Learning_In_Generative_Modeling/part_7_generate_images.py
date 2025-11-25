def generate_images(model, n_samples):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, 20)  # Sample from a standard normal distribution
        generated_images = model.decoder(z)
        return generated_images.view(-1, 1, 28, 28)

# Generate 10 new images
new_images = generate_images(model, 10)
