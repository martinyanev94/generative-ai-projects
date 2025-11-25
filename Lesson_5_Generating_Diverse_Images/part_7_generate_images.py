# Generate new images post-training
for _ in range(5):  # Generate 5 images
    latent_vector = torch.randn(1, 100)
    generated_image = generator(latent_vector)
    show_image(generated_image)
