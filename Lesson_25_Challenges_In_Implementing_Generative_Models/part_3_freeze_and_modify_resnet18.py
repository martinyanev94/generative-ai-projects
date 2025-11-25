import torchvision.models as models

pretrained_model = models.resnet18(pretrained=True)
# Freeze the parameters of the pre-trained model
for param in pretrained_model.parameters():
    param.requires_grad = False

# Modify the last layer to fit our dataset
pretrained_model.fc = torch.nn.Linear(pretrained_model.fc.in_features, num_classes)
