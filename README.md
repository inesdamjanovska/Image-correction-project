
# Adaptive Image Enhancement with Gamma and Beta Correction

A Python-based tool for adaptive image enhancement using dynamic gamma and brightness correction.

## Overview

Traditional methods like histogram equalization or fixed gamma correction lack adaptability across various lighting conditions. This project introduces an automated approach using adaptive gamma and beta values, based on brightness analysis.

## Features

- Dynamic brightness and contrast adjustment
- Automatic detection of lighting conditions (dark, normal, bright)
- Comparison with OpenCV’s fixed enhancement methods
- GUI interface built with Tkinter
- Supports .png, .jpg, .jpeg formats

## Setup

### (Optional but recommended) Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment:
- On Windows:  
  ```bash
  venv\Scripts\activate
  ```
- On Mac/Linux:  
  ```bash
  source venv/bin/activate
  ```

### Install the required packages

If you have a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, you can manually install the necessary packages:

```bash
pip install opencv-python numpy pillow
```

### Run via Command Line

To process a batch of images from a folder and save outputs:

```bash
python main.py
```

Make sure to edit the input/output folders in `main.py` if needed:

```python
IMAGE_DIR = "image_correction_project/images"
OUTPUT_DIR = "image_correction_project/output"
```

### Run the GUI

To launch the graphical interface and load, view, and compare enhanced images:

```bash
python gui.py
```
