# Load images
content_image = load_image("path_to_your_content_image.jpg")
style_image = load_image("path_to_your_style_image.jpg")

# Initialize generated image
generated_image = content_image.clone().requires_grad_(True)

# Set up the optimizer
optimizer = torch.optim.LBFGS([generated_image], lr=0.1)

model = models.vgg19(pretrained=True).features.eval()  # Load the VGG model
# Extract features for content and style
content_features = get_features(content_image, model)
style_features = get_features(style_image, model)

# Optimization loop
for i in range(1, 301):
    def closure():
        optimizer.zero_grad()
        target_features = get_features(generated_image, model)
        c_loss = content_loss(target_features, content_features)
        s_loss = style_loss(target_features, style_features)
        total_loss = 1e5 * s_loss + c_loss  # Adjust weight as necessary
        total_loss.backward()
        return total_loss

    optimizer.step(closure)
    if i % 50 == 0:
        print(f"Step {i}, Total Loss: {closure().item()}")
