import numpy as np

def evaluate_diversity(generated_images, num_samples=100):
    random_indices = np.random.choice(len(generated_images), num_samples, replace=False)
    sampled_images = [generated_images[i] for i in random_indices]
    
    # Assuming there's a method to compute feature vectors of images
    feature_vectors = compute_feature_vectors(sampled_images)
    diversity_metric = np.std(feature_vectors, axis=0)  # Measure of diversity

    return diversity_metric
