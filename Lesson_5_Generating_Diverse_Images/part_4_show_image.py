import matplotlib.pyplot as plt
import numpy as np

def show_image(tensor):
    # Convert tensor image to numpy array
    image = tensor.detach().numpy()
    image = (image + 1) / 2  # Rescale to [0, 1]
    image = np.transpose(image, (0, 2, 3, 1))  # Change from CxHxW to HxWxC
    plt.imshow(image[0])  # Show the first image in the batch
    plt.axis('off')
    plt.show()

show_image(generated_image)
