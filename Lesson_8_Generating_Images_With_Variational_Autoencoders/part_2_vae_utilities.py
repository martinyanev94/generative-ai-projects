def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)  # Calculate standard deviation
    eps = torch.randn_like(std)     # Generate random noise
    return mu + eps * std           # Return the sampled latent variable
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
