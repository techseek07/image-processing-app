# image_processing.py
import cv2
import numpy as np

def apply_smoothing(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def apply_log_transformation(image):
    image = np.array(image, dtype=np.float32) + 1  # Avoid log(0)
    log_transformed = np.log(image) * (255 / np.log(256))
    return np.array(np.clip(log_transformed, 0, 255), dtype=np.uint8)

def apply_histogram_equalization(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(image)

def apply_edge_detection(image):
    return cv2.Canny(image, 100, 200)
def increase_contrast(image, alpha=1.5, beta=0):
    # alpha: contrast control (1.0-3.0), beta: brightness control
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
def remove_noise(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


def apply_averaging_filter(image, kernel_size=3):
    """
    Applies an averaging filter to the image to perform low-pass filtering.

    Parameters:
    - image: Input image (numpy array).
    - kernel_size: Size of the square kernel; must be a positive odd integer.

    Returns:
    - Filtered image with reduced high-frequency components.
    """
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("Kernel size must be a positive odd integer.")

    # Create an averaging kernel
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)

    # Apply the averaging filter
    filtered_image = cv2.filter2D(image, -1, kernel)

    return filtered_image

def apply_high_pass_filter(image, kernel_size=3):
    """Applies a high-pass filter to the image to enhance edges."""
    # Create a high-pass filter kernel
    kernel = -1 * np.ones((kernel_size, kernel_size))
    kernel[kernel_size//2, kernel_size//2] = kernel_size * kernel_size - 1
    return cv2.filter2D(image, -1, kernel)