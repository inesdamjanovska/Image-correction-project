import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

# Ensure the src directory is in Python's path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'image_correction_project', 'src')))

from enhancement.opencv_enhancement import opencv_enhance
from enhancement.custom_enhancement import custom_enhance
from utils.image_processing import load_image, save_image
from utils.comparison import calculate_difference

def auto_adjust_gamma_beta(image):
    """Automatically adjusts gamma and beta based on brightness analysis."""
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)  # Calculate mean pixel intensity

    # Adjust brightness and contrast based on brightness level
    if mean_brightness < 100:  
        gamma = 1.8  
        beta = 50  
    elif mean_brightness > 180:  
        gamma = 0.7  
        beta = -30  
    else:  
        gamma = 1.2
        beta = 10  

    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    corrected = cv2.LUT(image, table)
    adjusted = cv2.convertScaleAbs(corrected, alpha=1, beta=beta)

    return adjusted

class ImageEnhancerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Enhancement Viewer")
        self.root.geometry("800x600")
        
        # UI Elements
        self.label = Label(root, text="Select an image to enhance")
        self.label.pack()

        self.btn_select = Button(root, text="Select Image", command=self.select_image)
        self.btn_select.pack()

        self.before_label = Label(root, text="Original Image")
        self.before_label.pack()
        self.before_canvas = tk.Canvas(root, width=300, height=300)
        self.before_canvas.pack()

        self.after_label = Label(root, text="Enhanced Image")
        self.after_label.pack()
        self.after_canvas = tk.Canvas(root, width=300, height=300)
        self.after_canvas.pack()

        self.result_label = Label(root, text="Difference Score: ")
        self.result_label.pack()

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if not file_path:
            return
        
        self.process_image(file_path)

    def process_image(self, file_path):
        # Load image
        image = load_image(file_path)
        if image is None:
            self.label.config(text="Error: Could not load image")
            return
        
        # Apply Auto Enhancement
        auto_result = auto_adjust_gamma_beta(image)

        # Apply OpenCV Enhancement
        opencv_result = opencv_enhance(image, alpha=1.5, beta=50)

        # Apply Custom Enhancement
        custom_result = custom_enhance(image)

        # Calculate Difference Score
        diff_score = calculate_difference(auto_result, custom_result)

        # Convert images for display
        original_img = self.convert_image(image)
        enhanced_img = self.convert_image(auto_result)

        # Update UI
        self.display_image(self.before_canvas, original_img)
        self.display_image(self.after_canvas, enhanced_img)
        self.result_label.config(text=f"Difference Score: {diff_score:.2f}")

    def convert_image(self, cv_image):
        """Convert OpenCV image to PIL format."""
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(cv_image)
        img_pil = img_pil.resize((300, 300), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img_pil)

    def display_image(self, canvas, img):
        """Display image on the given canvas."""
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        canvas.image = img  

# Run the UI
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEnhancerApp(root)
    root.mainloop()
