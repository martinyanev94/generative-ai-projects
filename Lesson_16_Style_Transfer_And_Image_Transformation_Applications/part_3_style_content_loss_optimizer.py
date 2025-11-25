def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G.div(c * h * w)

def get_style_content_loss(style, content, target_style, target_content):
    content_loss = nn.MSELoss()(content, target_content)
    style_loss = sum(nn.MSELoss()(gram_matrix(s), g) for s, g in zip(style, target_style))
    total_loss = content_loss + style_loss
    return total_loss
target_image = content_image.clone().requires_grad_(True)
optimizer = torch.optim.Adam([target_image], lr=0.003)

for i in range(1000):
    optimizer.zero_grad()
    
    target_content, target_style = vgg(target_image)
    content_loss = nn.MSELoss()(target_content[2], vgg(content_image)[0][2])
    style_loss = sum(nn.MSELoss()(gram_matrix(target_style[j]), gram_matrix(vgg(style_image)[1][j])) for j in range(len(target_style)))
    
    total_loss = content_loss + style_loss
    total_loss.backward()
    optimizer.step()
    
    if i % 100 == 0:
        print('Iteration', i, 'Total loss:', total_loss.item())
