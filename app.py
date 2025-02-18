import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from pdf2image import convert_from_path
import os
import tempfile

# Define the model file path
current_directory = os.path.dirname(__file__)
model_file_path = os.path.join(current_directory, 'pneumonia_model.h5')

# Check if the model file exists
if not os.path.exists(model_file_path):
    st.error(f"Model file not found at {model_file_path}")
else:
    # Load the model
    model = tf.keras.models.load_model(model_file_path)

    # Introduction
    st.markdown("# Pneumonia Detection")
    st.markdown("## Introduction")
    st.markdown("Pneumonia is a serious lung infection that can be caused by bacteria, viruses, or fungi. Early detection is crucial for effective treatment. This app allows users to upload CT scan images for pneumonia detection using a trained deep learning model. Our goal is to assist healthcare professionals in diagnosing pneumonia accurately and efficiently.")
    st.markdown("## How it Works")
    st.markdown("1. Upload a CT scan image")
    st.markdown("2. Our model will analyze the image and predict whether it shows signs of pneumonia")
    st.markdown("3. The result will be displayed on the screen")

    # Streamlit App
    st.title("Pneumonia Detection from CT Scans")
    
    uploaded_file = st.file_uploader("Upload a CT scan image...", type=["jpeg", "jpg", "png", "pdf"])

    def preprocess_image(image_file):
        """Preprocess image for model prediction"""
        if image_file.type == "application/pdf":
            # Save uploaded PDF temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(image_file.read())
                temp_pdf_path = temp_pdf.name

            # Convert first page of PDF to image
            images = convert_from_path(temp_pdf_path)
            image = np.array(images[0])  # Convert PIL image to NumPy array

        else:
            # Open image using PIL
            image = Image.open(image_file)
            image = np.array(image)  # Convert to NumPy array

        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:  # If grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Resize to match model input
        image = cv2.resize(image, (150, 150))

        # Expand dimensions for model input (1, 150, 150, 3)
        image = np.expand_dims(image, axis=0)

        # Normalize pixel values
        return image / 255.0

    if uploaded_file is not None:
        image = preprocess_image(uploaded_file)

        # Make prediction
        prediction = model.predict(image)

        # Display result
        if prediction < 0.5:
            st.success("✅ The scan is **Normal**")
        else:
            st.error("⚠️ The scan shows **Pneumonia**")

        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded CT Scan", use_container_width=True)
