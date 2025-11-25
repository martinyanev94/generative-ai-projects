from itertools import product

# Learning rates to try
learning_rates = [0.0001, 0.0002, 0.0005]
results = {}

for lr in learning_rates:
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
    g_optimizer = optim.Adam(generator.parameters(), lr=lr)
    train_gan(generator, discriminator, data_loader, epochs=10)
    results[lr] = evaluate_model(generator)  # Assuming evaluate_model function exists
