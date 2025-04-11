import cv2
import os
from src.enhancement.opencv_enhancement import opencv_enhance
from src.enhancement.custom_enhancement import custom_enhance
from src.utils.image_utils import load_image, save_image
import numpy as np

def auto_adjust_gamma_beta(image):
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

# Define paths
IMAGE_DIR = "image_correction_project/images"
OUTPUT_DIR = "image_correction_project/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List of images to process
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(IMAGE_DIR, image_file)
    image = load_image(image_path)
    
    if image is None:
        print(f"Error: Could not load {image_file}")
        continue

    # Apply automatic enhancement
    auto_result = auto_adjust_gamma_beta(image)
    save_image(auto_result, os.path.join(OUTPUT_DIR, f"auto_{image_file}"))

    # Apply OpenCV enhancement
    opencv_result = opencv_enhance(image, alpha=1.2, beta=30)
    save_image(opencv_result, os.path.join(OUTPUT_DIR, f"opencv_{image_file}"))
    
    # Apply custom enhancement
    custom_result = custom_enhance(image, gamma=0.9, clip_limit=1.5)
    save_image(custom_result, os.path.join(OUTPUT_DIR, f"custom_{image_file}"))

    # Show results
    cv2.imshow("Original Image", image)
    cv2.imshow("Auto Enhanced", auto_result)
    cv2.imshow("OpenCV Enhanced", opencv_result)
    cv2.imshow("Custom Enhanced", custom_result)

    print(f"Processed {image_file} and saved results in output folder.")

cv2.waitKey(0)
cv2.destroyAllWindows()
