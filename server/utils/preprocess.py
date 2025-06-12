import cv2
import numpy as np

def preprocess_image(image):
    """
    Preprocess the input image to match the model's input requirements.
    """
    # Resize the image to the input size of the model
    image = cv2.resize(image, (64, 64))
    
    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normalize the image
    image = image / 255.0
    
    # Expand dimensions to match the model input shape
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    
    return image