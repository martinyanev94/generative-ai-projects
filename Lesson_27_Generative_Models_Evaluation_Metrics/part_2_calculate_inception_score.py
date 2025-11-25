import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import Model

def calculate_inception_score(generated_images):
    model = InceptionV3()  # Load a pre-trained Inception model
    scores = []
    
    for img in generated_images:
        img = preprocess_input(img)  # Preprocess the image for Inception
        img_array = img_to_array(img)  # Convert the image to an array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        preds = model.predict(img_array)  # Predict the class probabilities
        scores.append(preds)

    scores = np.array(scores)
    # Calculate the mean and variance of the predicted probabilities
    mean_scores = np.mean(scores, axis=0)
    variance_scores = np.var(scores, axis=0)
    inception_score = np.exp(np.mean(np.log(mean_scores)) - np.mean(np.log(variance_scores)))
    
    return inception_score
