from scipy.linalg import sqrtm
import numpy as np

def calculate_fid(real_samples, generated_samples):
    mu_real, sigma_real = np.mean(real_samples, axis=0), np.cov(real_samples, rowvar=False)
    mu_gen, sigma_gen = np.mean(generated_samples, axis=0), np.cov(generated_samples, rowvar=False)

    fid = np.sum((mu_real - mu_gen) ** 2) + np.trace(sigma_real + sigma_gen - 2 * sqrtm(sigma_real @ sigma_gen))
    return fid
