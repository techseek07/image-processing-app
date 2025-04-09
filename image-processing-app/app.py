import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Image Processing Functions
def apply_smoothing(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def apply_log_transformation(image):
    image = np.array(image, dtype=np.float32) + 1  # Avoid log(0)
    log_transformed = np.log(image) * (255 / np.log(256))

    log_transformed = np.clip(log_transformed, 0, 255)  # Ensure valid range
    return np.array(log_transformed, dtype=np.uint8)


def apply_histogram_equalization(image):
    # Convert to grayscale if the image has multiple channels
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return cv2.equalizeHist(image)

def apply_edge_detection(image):
    return cv2.Canny(image, 100, 200)

# Streamlit UI
st.title("Image Enhancement & Processing")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)

    st.image(image, caption="Original Image", use_column_width=True)

    option = st.selectbox("Choose an enhancement technique",
                          ["Smoothing", "Log Transformation", "Histogram Equalization", "Edge Detection"])

    if st.button("Apply"):
        if option == "Smoothing":
            processed_image = apply_smoothing(image)
        elif option == "Log Transformation":
            processed_image = apply_log_transformation(image)
        elif option == "Histogram Equalization":
            processed_image = apply_histogram_equalization(image)
        elif option == "Edge Detection":
            processed_image = apply_edge_detection(image)

        st.image(processed_image, caption="Processed Image", use_column_width=True, channels="GRAY")