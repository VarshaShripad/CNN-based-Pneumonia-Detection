import os
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from pdf2image import convert_from_path
import tempfile

# Paths
data_dir = r"D:\Varsha\AIML"  # Update your dataset path
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
model_path = os.path.join(data_dir, "pneumonia_model.h5")

# Step 1: Split Dataset into Train and Test
def split_dataset():
    # Create train and test directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for category in ["non-covid", "covid"]:
        category_path = os.path.join(data_dir, category)

        if not os.path.exists(category_path):
            st.error(f"Category folder not found: {category_path}")
            return

        # Get all images in the category folder
        images = os.listdir(category_path)

        if len(images) == 0:
            st.error(f"No images found in {category_path}")
            return

        # Split images into train and test sets
        train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

        # Create subfolders for each category in train and test directories
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(test_dir, category), exist_ok=True)

        # Move images to the train and test folders
        for image in train_images:
            src = os.path.join(category_path, image)
            dst = os.path.join(train_dir, category, image)
            if os.path.exists(src):
                shutil.move(src, dst)
                st.write(f"Moved {image} to {dst}")
            else:
                st.error(f"Image {src} not found!")

        for image in test_images:
            src = os.path.join(category_path, image)
            dst = os.path.join(test_dir, category, image)
            if os.path.exists(src):
                shutil.move(src, dst)
                st.write(f"Moved {image} to {dst}")
            else:
                st.error(f"Image {src} not found!")

    st.write("Dataset splitting complete.")

# Step 2: Train the Model
def train_model():
    train_datagen = ImageDataGenerator(rescale=1.0/255.0, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    val_generator = val_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_generator, epochs=10, validation_data=val_generator)
    model.save(model_path)

    st.write("Model trained and saved.")

# Step 3: Streamlit Application
def run_app():
    st.title("Pneumonia Detection from CT Scans")

    if not os.path.exists(model_path):
        st.error("Model not found. Please train the model first.")
        return

    model = tf.keras.models.load_model(model_path)

    uploaded_file = st.file_uploader("Upload a CT scan image...", type=["jpeg", "jpg", "png", "pdf"])

    def preprocess_image(image):
        if image.type == "application/pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(image.read())
                images = convert_from_path(temp_pdf.name)
            image = np.array(images[0])
        else:
            image = Image.open(image)
            image = np.array(image)

    # Resize the image to 150x150
        image = cv2.resize(image, (150, 150))

    # Ensure the image has 3 channels (RGB)
        if len(image.shape) == 2:  # Grayscale image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Normalize and expand dimensions
        image = image / 255.0
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image

    if uploaded_file is not None:
        image = preprocess_image(uploaded_file)
        prediction = model.predict(image)
        if prediction < 0.5:
            st.write("The scan is Normal")
        else:
            st.write("The scan shows Pneumonia")

        st.image(uploaded_file, caption="Uploaded CT Scan", use_column_width=True)

# Streamlit UI
st.sidebar.title("Options")
task = st.sidebar.radio("Select Task", ["Split Dataset", "Train Model", "Run App"])

if task == "Split Dataset":
    split_dataset()
elif task == "Train Model":
    train_model()
elif task == "Run App":
    run_app()
