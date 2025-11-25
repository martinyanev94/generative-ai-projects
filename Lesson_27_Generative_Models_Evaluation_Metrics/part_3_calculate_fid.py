import tensorflow as tf
from scipy.linalg import sqrtm

def calculate_fid(real_images, generated_images):
    model = InceptionV3()
    
    # Get activations for real and generated images
    real_activations = model.predict(real_images)
    generated_activations = model.predict(generated_images)
    
    # Calculate the mean and covariance
    mu_real, cov_real = np.mean(real_activations, axis=0), np.cov(real_activations, rowvar=False)
    mu_gen, cov_gen = np.mean(generated_activations, axis=0), np.cov(generated_activations, rowvar=False)
    
    # Calculate the squared difference in means
    mean_diff = mu_real - mu_gen
    ssdiff = np.sum(mean_diff ** 2)
    
    # Calculate the FID score
    cov_sqrt = sqrtm(cov_real.dot(cov_gen))
    fid = ssdiff + np.trace(cov_real + cov_gen - 2 * cov_sqrt)
    
    return fid
