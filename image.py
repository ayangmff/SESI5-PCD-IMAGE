import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'C:\\Users\\lenovo\\Downloads\\LAUTTT.jpg'  # Ganti dengan path gambar Anda
image = cv2.imread(image_path)

# Check if the image was loaded
if image is None:
    print(f"Gambar tidak ditemukan di path: {image_path}")
else:
    # Convert the image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Function to display images
    def display_images(images, titles, cmap=None):
        plt.figure(figsize=(15, 5))
        for i in range(len(images)):
            plt.subplot(1, len(images), i + 1)
            plt.imshow(images[i], cmap=cmap)
            plt.title(titles[i])
            plt.axis('off')
        plt.show()

    # 1. Low-Pass Filter (Gaussian Blur) - Smoothing effect
    def apply_low_pass_filter(image):
        return cv2.GaussianBlur(image, (11, 11), 0)

    # 2. High-Pass Filter (Laplacian) - Edge detection
    def apply_high_pass_filter(image):
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return cv2.convertScaleAbs(laplacian)

    # 3. High-Boost Filter - Enhanced sharpening effect
    def apply_high_boost_filter(image, boost_factor=1.5):
        blurred = cv2.GaussianBlur(image, (11, 11), 0)
        high_boost = cv2.addWeighted(image, boost_factor, blurred, -0.5, 0)
        return high_boost

    # Apply filters on color image
    color_low_pass = apply_low_pass_filter(image)
    color_high_pass = apply_high_pass_filter(image)
    color_high_boost = apply_high_boost_filter(image)

    # Apply filters on grayscale image
    gray_low_pass = apply_low_pass_filter(image_gray)
    gray_high_pass = apply_high_pass_filter(image_gray)
    gray_high_boost = apply_high_boost_filter(image_gray)

    # Display results for color image
    display_images(
        [cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
         cv2.cvtColor(color_low_pass, cv2.COLOR_BGR2RGB), 
         cv2.cvtColor(color_high_pass, cv2.COLOR_BGR2RGB), 
         cv2.cvtColor(color_high_boost, cv2.COLOR_BGR2RGB)],
        ["Original Color", "Low-Pass Filter (Color)", "High-Pass Filter (Color)", "High-Boost Filter (Color)"]
    )

    # Display results for grayscale image
    display_images(
        [image_gray, gray_low_pass, gray_high_pass, gray_high_boost],
        ["Original Grayscale", "Low-Pass Filter (Grayscale)", "High-Pass Filter (Grayscale)", "High-Boost Filter (Grayscale)"],
        cmap='gray'
    )
