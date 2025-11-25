import torch
from torchvision import models

def apply_gradcam(model, input_image, target_class):
    model.eval()
    output = model(input_image)
    
    one_hot_output = torch.zeros((1, output.size()[-1]), dtype=torch.float32)
    one_hot_output[0][target_class] = 1
    model.zero_grad()
    output.backward(gradient=one_hot_output)

    # Generate heatmap
    gradients = model.get_activations_gradient().detach().cpu().numpy()
    activation = model.get_activations(input_image).detach().cpu().numpy()
    weights = np.mean(gradients, axis=(2, 3))[0, :]
    
    heatmap = np.zeros(activation.shape[2:], dtype=np.float32)
    for i in range(weights.shape[0]):
        heatmap += weights[i] * activation[0, i, :, :]

    return heatmap
