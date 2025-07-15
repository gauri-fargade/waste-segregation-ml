import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os

# Loading the trained model
model = tf.keras.models.load_model("waste_classification_model.h5")

# Defining class labels
class_labels = {0: "Organic", 1: "Reusable"}

# Function to preprocess an image
def preprocess_image(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Streamlit UI
st.set_page_config(page_title="Waste Classification", layout="centered")
st.title("♻️ Waste Classification System")
st.subheader("**Sort Smart, Save Earth! 🌍**")
st.write("Upload an image to classify it as Organic or Reusable.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    
    file_path = "temp.jpg"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
   
    image_data = preprocess_image(file_path)
    prediction = model.predict(image_data)
    predicted_class = 1 if prediction[0][0] > 0.5 else 0
    class_name = class_labels[predicted_class]
    
   
    st.image(file_path, caption=f"Uploaded Image", use_column_width=True)
    st.markdown(f"### **Prediction: {class_name}**")
    
   
    os.remove(file_path)


