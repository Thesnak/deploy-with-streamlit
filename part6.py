import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import os

# --- 1. Save the model --- 
# Define a directory to save the model
model_save_path = "models/bird_classifier_model.keras"

# --- 2. Preprocessing function ---
def preprocess_image(image):
    # Resize the image to the target size (128, 128)
    image = image.resize((128, 128))
    # Convert PIL Image to NumPy array
    image = np.array(image)
    # The model's first layer is Rescaling(1./255), so input should be 0-255
    # Expand dimensions to create a batch of 1 image
    image = np.expand_dims(image, axis=0)
    return image

# --- 3. Save some example images for uploading ---
# --- 4. Streamlit App --- 

st.title("Bird Species Classifier")
st.write("Upload an image of a bird to classify its species!")

# Load the trained model
@st.cache_resource
def load_my_model():
    model = keras.models.load_model(model_save_path)
    return model

model_loaded = load_my_model()

# Get class names from the training dataset
class_names = ['AMERICAN GOLDFINCH',
 'BARN OWL',
 'CARMINE BEE-EATER',
 'DOWNY WOODPECKER',
 'EMPEROR PENGUIN',
 'FLAMINGO']

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image')
    st.write("")
    st.write("Classifying...")

    # Preprocess and predict
    processed_image = preprocess_image(image)
    predictions = model_loaded.predict(processed_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions, axis=1)[0]

    predicted_class_name = class_names[predicted_class_index]

    st.success(f"Prediction: **{predicted_class_name}** with a confidence of **{confidence:.2f}**")

st.write("You can find example images in the './example_bird_images' directory to test the app.")
st.write("To run this Streamlit app, save this code as a Python file (e.g., `app.py`) and run `streamlit run app.py` in your terminal.")