import cv2

def load_image(image_path):
    """Loads an image from a given path."""
    return cv2.imread(image_path)

def save_image(image, save_path):
    """Saves an image to a given path."""
    cv2.imwrite(save_path, image)
