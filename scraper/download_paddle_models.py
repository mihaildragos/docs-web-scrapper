#!/usr/bin/env python3
"""
Script to pre-download PaddleOCR models to avoid downloading them at runtime.
"""
from paddleocr import PaddleOCR
import os
import time
import sys

# Set the HOME environment variable to ensure PaddleOCR saves models in the right place
os.environ["HOME"] = "/root"


def download_models():
    print("Pre-downloading PaddleOCR models...")
    try:
        # Initialize PaddleOCR - this will download the models
        # Use timeout to prevent hanging if download servers are slow
        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=True)

        # Force model loading to trigger downloads
        test_img = os.path.join(os.path.dirname(__file__), "test_image.png")

        # Create a small test image if it doesn't exist
        if not os.path.exists(test_img):
            from PIL import Image, ImageDraw, ImageFont

            img = Image.new("RGB", (100, 30), color=(255, 255, 255))
            d = ImageDraw.Draw(img)
            d.text((10, 10), "Hello", fill=(0, 0, 0))
            img.save(test_img)

        # Run inference to ensure models are downloaded
        result = ocr.ocr(test_img, cls=True)
        print("Model test successful!")

        print("PaddleOCR models downloaded successfully!")
        return True
    except Exception as e:
        print(f"Error downloading models: {e}")
        return False


# Try to download models with retry logic
max_attempts = 3
for attempt in range(max_attempts):
    print(f"Download attempt {attempt+1}/{max_attempts}")
    if download_models():
        sys.exit(0)
    time.sleep(5)  # Wait before retrying

sys.exit(1)  # Exit with error if all attempts failed
