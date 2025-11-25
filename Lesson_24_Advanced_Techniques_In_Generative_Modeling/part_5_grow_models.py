def grow_models(generator, discriminator):
    # Define the layers for the new resolution
    new_layers_gen = nn.Sequential(
        nn.Linear(128, 512),
        nn.ReLU(),
        nn.Linear(512, 8 * 8),
        nn.Tanh()
    )
    
    new_layers_disc = nn.Sequential(
        nn.Linear(8 * 8, 512),
        nn.LeakyReLU(0.2),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )
    # Add new layers to both models
    generator.add_module('growth', new_layers_gen)
    discriminator.add_module('growth', new_layers_disc)
