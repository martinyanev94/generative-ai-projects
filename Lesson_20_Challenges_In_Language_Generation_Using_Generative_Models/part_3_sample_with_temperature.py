import random

def sample_with_temperature(prediction_distribution, temperature=1.0):
    prediction_distribution = prediction_distribution / np.sum(prediction_distribution)
    prediction_distribution = np.exp(np.log(prediction_distribution) / temperature)
    prediction_distribution = prediction_distribution / np.sum(prediction_distribution)
    return np.random.choice(range(len(prediction_distribution)), p=prediction_distribution)

# Example: Assuming we have a simulated prediction distribution from the model
dummy_prediction_distribution = np.array([0.1, 0.2, 0.3, 0.4])
next_word = sample_with_temperature(dummy_prediction_distribution, temperature=0.7)
print(f"Next word choice based on sampling: {next_word}")
