def content_loss(target_features, content_features):
    return torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G.div(c * h * w)

def style_loss(target_features, style_features):
    loss = 0.0
    layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1']
    for layer in layers:
        target_gram = gram_matrix(target_features[layer])
        style_gram = gram_matrix(style_features[layer])
        loss += torch.mean((target_gram - style_gram) ** 2)
    return loss
