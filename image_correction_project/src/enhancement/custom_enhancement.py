import cv2
import numpy as np

def custom_enhance(image):
    """Automatically adjusts gamma and beta based on brightness analysis."""
    
    # Convert to grayscale to analyze brightness
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)  # Calculate mean pixel intensity

    # Define gamma and beta adjustments based on brightness levels
    if mean_brightness < 100:  # Too dark
        gamma = 1.8  # Increase contrast
        beta = 50  # Increase brightness
    elif mean_brightness > 180:  # Too bright
        gamma = 0.7  # Reduce contrast
        beta = -30  # Reduce brightness
    else:  # Normal
        gamma = 1.2
        beta = 10

    # Apply gamma correction
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    corrected = cv2.LUT(image, table)

    # Apply brightness adjustment
    adjusted = cv2.convertScaleAbs(corrected, alpha=1, beta=beta)

    return adjusted

