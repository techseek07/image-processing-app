import streamlit as st
import numpy as np
from PIL import Image
import cv2

from image_processing import (
    apply_smoothing, apply_log_transformation,
    apply_histogram_equalization, apply_edge_detection,
    increase_contrast, remove_noise,
    apply_averaging_filter as apply_low_pass_filter,  # renamed from Gaussian
    apply_high_pass_filter, sharpen_image
)

# Optional: Configure Streamlit page properties
st.set_page_config(page_title="Image Enhancement Tools", layout="wide")


@st.cache_data
def load_image(image_file):
    return Image.open(image_file)



def main():
    st.title("🖼️ Image Enhancement & Processing App")
    st.markdown("Upload an image and select an enhancement technique from the sidebar.")

    # Sidebar: Upload and options
    st.sidebar.title("Upload & Options")
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    # Enhancement options
    enhancement_options = [
        "Smoothing", "Log Transformation", "Histogram Equalization", "Edge Detection",
        "Increase Contrast", "Remove Noise", "Low-Pass Filtering", "High-Pass Filtering", "Sharpen Image"
    ]
    option = st.sidebar.selectbox("Choose an enhancement technique", enhancement_options)

    # Parameter controls for techniques that need them
    params = {}
    if option == "Increase Contrast":
        params['alpha'] = st.sidebar.slider("Contrast (alpha)", 1.0, 3.0, 1.5)
        params['beta'] = st.sidebar.slider("Brightness (beta)", 0, 100, 0)
    if option == "Remove Noise":
        params['kernel_size'] = st.sidebar.slider("Kernel Size", 3, 11, 5, step=2)
    if option in ["Low-Pass Filtering", "High-Pass Filtering"]:
        params['kernel_size'] = st.sidebar.slider("Kernel Size", 1, 10, 3)

    # Process image if uploaded
    if uploaded_file is not None:
        original_image = load_image(uploaded_file)
        image_array = np.array(original_image)

        # Apply button
        if st.sidebar.button("Apply Enhancement"):
            with st.spinner("Processing image..."):
                processed_image = None
                try:
                    if option == "Smoothing":
                        processed_image = apply_smoothing(image_array)
                    elif option == "Log Transformation":
                        processed_image = apply_log_transformation(image_array)
                    elif option == "Histogram Equalization":
                        processed_image = apply_histogram_equalization(image_array)
                    elif option == "Edge Detection":
                        processed_image = apply_edge_detection(image_array)
                    elif option == "Increase Contrast":
                        processed_image = increase_contrast(image_array, alpha=params['alpha'], beta=params['beta'])
                    elif option == "Remove Noise":
                        processed_image = remove_noise(image_array, kernel_size=params['kernel_size'])
                    elif option == "Low-Pass Filtering":
                        processed_image = apply_low_pass_filter(image_array, kernel_size=params['kernel_size'])
                    elif option == "High-Pass Filtering":
                        processed_image = apply_high_pass_filter(image_array, kernel_size=params['kernel_size'])
                    elif option == "Sharpen Image":
                        processed_image = sharpen_image(image_array)
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")

            # Display original and processed images
            if processed_image is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.header("Original Image")
                    st.image(original_image, use_column_width=True)
                with col2:
                    st.header("Enhanced Image")
                    if len(processed_image.shape) == 2:
                        st.image(processed_image, caption="Processed Image", use_column_width=True, channels="GRAY")
                    else:
                        st.image(processed_image, caption="Processed Image", use_column_width=True)
            else:
                st.error("Processing failed. Please try a different technique or adjust parameters.")
    else:
        st.info("Please upload an image to get started.")


if __name__ == '__main__':
    main()
