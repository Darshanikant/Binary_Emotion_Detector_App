import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import streamlit as st

# Load the trained model
model =tf.keras.models.load_model("emotion_detector_model.h5")
# Fix random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Function to predict image class
def predict_image_class(image, model):
    
    img_array = img_to_array(image) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)[0][0]

    # Interpret result
    return "Happy" if prediction < 0.5 else "Sad"

# Streamlit app
st.title("Image Detector App (Happy or Sad)")

st.markdown(
    """
    Welcome to the **Emotion Detector App**. Upload your images to predict if they convey happiness or sadness.
    This app uses a **Convolutional Neural Network (CNN)** model to perform binary classification. 
    Created with ❤️ by **Darshanikanta**.
    """
)
st.caption("© 2024 Darshanikanta. All rights reserved.")


uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])


if uploaded_files:
    st.subheader("Uploaded Images:")
    for uploaded_file in uploaded_files:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=False)


if st.button("Detect"):
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Load image
            image = load_img(uploaded_file, target_size=(200, 200))
            #model_fit=model.fit(image,epochs=50)

            # Predict class
            result = predict_image_class(image, model)

            # Display image and result
           # st.image(image,caption=f"Prediction: {result}", use_column_width=True)
            st.subheader(f"Its a {result} Image")
    else:
        st.warning("Please upload at least one image.")
