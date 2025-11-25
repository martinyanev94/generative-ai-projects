def cycle_loss(real_image, cycled_image, lambda_cycle=10):
    return nn.L1Loss()(real_image, cycled_image) * lambda_cycle
