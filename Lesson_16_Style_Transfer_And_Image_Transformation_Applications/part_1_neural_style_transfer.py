import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(img_path, max_size=400):
    image = Image.open(img_path)
    size = max(image.size)
    if size > max_size:
        image = transforms.Resize(max_size)(image)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.unsqueeze(0)),
    ])
    return transform(image).to(device)

style_image = load_image('path_to_style_image.jpg')
content_image = load_image('path_to_content_image.jpg')
class VGG(nn.Module):
    def __init__(self, layers):
        super(VGG, self).__init__()
        self.model = models.vgg19(pretrained=True).features[:max(layers)].to(device).eval()
    
    def forward(self, x):
        content = []
        style = []
        for layer in self.model.children():
            x = layer(x)
            if layer.__class__.__name__ == 'Conv2d':
                style.append(x)
            if layer.__class__.__name__ == 'ReLU':
                content.append(x)
        return content, style

vgg = VGG(layers=[0, 5, 10, 19, 28])
