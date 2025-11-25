def generator_loss(fake_output):
    return nn.BCELoss()(fake_output, torch.ones_like(fake_output))

def discriminator_loss(real_output, fake_output):
    real_loss = nn.BCELoss()(real_output, torch.ones_like(real_output))
    fake_loss = nn.BCELoss()(fake_output, torch.zeros_like(fake_output))
    return (real_loss + fake_loss) / 2
