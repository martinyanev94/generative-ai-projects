from torchvision.models import resnet18

# Load a pre-trained ResNet model
pretrained_model = resnet18(pretrained=True)

# Freeze the layers if we want to only train the final layer
for param in pretrained_model.parameters():
    param.requires_grad = False

# Replace the final classification layer with your specific number of classes
# Note: This is an illustration; adapt to your use case accordingly.
pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, <num_classes>)
