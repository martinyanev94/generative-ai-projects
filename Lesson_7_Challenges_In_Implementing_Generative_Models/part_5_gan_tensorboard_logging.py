from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/gan_experiment')

# During the training loop
for epoch in range(num_epochs):
    # ...train the GAN...
    writer.add_scalar('Generator Loss', g_loss.item(), epoch)
    writer.add_scalar('Discriminator Loss', d_loss.item(), epoch)
