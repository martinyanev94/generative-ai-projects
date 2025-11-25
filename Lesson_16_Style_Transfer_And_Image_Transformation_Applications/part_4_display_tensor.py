def imshow(tensor):
    tensor = tensor.detach().cpu().squeeze(0).clamp(0, 1)
    np_image = tensor.numpy()
    plt.imshow(np.transpose(np_image, (1, 2, 0)))
    plt.axis('off')
    plt.show()

imshow(target_image)
