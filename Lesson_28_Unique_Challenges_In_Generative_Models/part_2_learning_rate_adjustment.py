def adjust_learning_rate(optimizer, epoch, initial_lr=0.0002, decay_factor=0.1):
    """Decays the learning rate at certain epochs."""
    if epoch % 50 == 0 and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_factor
real_labels = torch.ones(batch_size, 1) * 0.9  # Label smoothing
fake_labels = torch.zeros(batch_size, 1) # Normal
