
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model('final_eco_model.keras')
class_names = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

st.title("EcoFriend Waste Classifier")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((224, 224))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    pred = model.predict(img_array)
    st.write(f"### Prediction: {class_names[np.argmax(pred)]}")
    st.image(image)
