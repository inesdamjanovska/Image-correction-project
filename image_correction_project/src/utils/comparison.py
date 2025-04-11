import cv2
import numpy as np

def calculate_difference(image1, image2):
    """Calculates the Mean Squared Error (MSE) between two images."""
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same shape for comparison.")

    mse = np.mean((image1.astype("float") - image2.astype("float")) ** 2)
    return mse

def compare_images(image1, image2):
    """Compares two images side by side for visual analysis."""
    return np.hstack((image1, image2))  
