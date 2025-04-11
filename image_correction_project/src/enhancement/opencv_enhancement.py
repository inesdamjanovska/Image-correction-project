import cv2
import numpy as np

def opencv_enhance(image, alpha=1.2, beta=30):
    """Enhances an image using OpenCV's contrast and brightness adjustment."""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
