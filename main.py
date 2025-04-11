import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from src.enhancement.opencv_enhancement import opencv_enhance
from src.enhancement.custom_enhancement import custom_enhance
from src.utils.image_utils import load_image, save_image

# === Метрики ===
def calculate_psnr(original, processed):
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_ssim(original, processed):
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(original_gray, processed_gray, full=True)
    return score

# === Колаж с PSNR и SSIM ===
def create_comparison_image(original, enhanced, label, psnr, ssim_score):
    annotated = enhanced.copy()
    cv2.putText(annotated, f"{label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(annotated, f"PSNR: {psnr:.2f} dB", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated, f"SSIM: {ssim_score:.4f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return np.hstack((original, annotated))

# === Динамична корекция ===
def auto_adjust_gamma_beta(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)

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

# === Пътища ===
IMAGE_DIR = "image_correction_project/images"
OUTPUT_DIR = "image_correction_project/output"
COMPARE_DIR = "image_correction_project/output/comparisons"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(COMPARE_DIR, exist_ok=True)

image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(IMAGE_DIR, image_file)
    image = load_image(image_path)

    if image is None:
        print(f"Error: Could not load {image_file}")
        continue

    # === Подобрения ===
    auto_result = auto_adjust_gamma_beta(image)
    opencv_result = opencv_enhance(image, alpha=1.2, beta=30)
    custom_result = custom_enhance(image, gamma=0.9, clip_limit=1.5)

    # === Запис на резултатите ===
    save_image(auto_result, os.path.join(OUTPUT_DIR, f"auto_{image_file}"))
    save_image(opencv_result, os.path.join(OUTPUT_DIR, f"opencv_{image_file}"))
    save_image(custom_result, os.path.join(OUTPUT_DIR, f"custom_{image_file}"))

    # === Метрики и колажи ===
    for method_name, result_img in [
        ("Auto", auto_result),
        ("OpenCV", opencv_result),
        ("Custom", custom_result)
    ]:
        psnr_val = calculate_psnr(image, result_img)
        ssim_val = calculate_ssim(image, result_img)
        comparison = create_comparison_image(image, result_img, method_name, psnr_val, ssim_val)
        save_image(comparison, os.path.join(COMPARE_DIR, f"{method_name.lower()}_comparison_{image_file}"))

    print(f"Processed {image_file} and saved results in output folder.")

cv2.destroyAllWindows()
