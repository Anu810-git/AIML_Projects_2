# app.py
# Streamlit app for my Brain Tumor MRI Image Classification project
# This just loads the model I trained in my notebook and lets a user upload
# an MRI image to get a prediction. Kept it simple since I'm still learning Streamlit :)

import streamlit as st                      # for building the web app
import numpy as np                           # for array handling
from tensorflow.keras.models import load_model   # to load my saved .h5 model
from PIL import Image                        # to open/resize the uploaded image
import pandas as pd                          # just to show the confidence scores as a small table

# ---------------------------------------------------------
# basic page settings - has to be the first streamlit command
st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="centered")

# ---------------------------------------------------------
# these are the 4 classes my model was trained on
# (order matters! this has to match train_generator.class_indices from my notebook)
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

IMG_SIZE = (224, 224)   # same size I used while training

# path to the model file I saved from my notebook
# I picked the transfer learning model since it had better accuracy in my testing
MODEL_PATH = "transfer_learning_model.h5"


# ---------------------------------------------------------
# loading the model only once and caching it, so the app doesn't reload it every time
# a user interacts with something on the page (that would be really slow)
@st.cache_resource
def get_model():
    model = load_model(MODEL_PATH)
    return model


def preprocess_image(image):
    """Takes a PIL image, resizes + normalizes it so it's ready for the model."""
    image = image.convert("RGB")                 # just in case the upload is grayscale or has alpha channel
    image = image.resize(IMG_SIZE)                # resize to the size the model expects
    img_array = np.array(image) / 255.0            # normalize pixel values (same as training)
    img_array = np.expand_dims(img_array, axis=0)   # model expects a "batch", so add an extra dimension
    return img_array


# ---------------------------------------------------------
# App title and small description
st.title("🧠 Brain Tumor MRI Image Classifier")
st.write(
    "Upload a brain MRI scan below and this app will try to predict the tumor type "
    "using the deep learning model I trained (transfer learning with MobileNetV2)."
)

st.info(
    "⚠️ Disclaimer: This is a student/learning project, NOT a medical diagnostic tool. "
    "Please do not use this for any real medical decisions."
)

# sidebar with a bit of project info, just for context
with st.sidebar:
    st.header("About this project")
    st.write(
        "This app is the deployment part of my Brain Tumor MRI Image Classification capstone project. "
        "The model was trained on brain MRI images to classify them into 4 categories:"
    )
    st.markdown("- Glioma\n- Meningioma\n- Pituitary\n- No Tumor")
    st.write("Built with TensorFlow/Keras + Streamlit.")


# ---------------------------------------------------------
# file uploader widget
uploaded_file = st.file_uploader(
    "Choose an MRI image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # showing the image the user uploaded
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    # simple button so prediction only happens when the user is ready
    if st.button("Predict Tumor Type"):

        with st.spinner("Loading model and running prediction..."):
            model = get_model()
            processed_img = preprocess_image(image)

            # getting the prediction probabilities for all 4 classes
            predictions = model.predict(processed_img)[0]

            predicted_index = np.argmax(predictions)          # index of the highest probability
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = predictions[predicted_index] * 100     # convert to percentage

        # showing the main result
        st.success(f"Prediction: **{predicted_class.upper()}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

        # a small extra message depending on the result, just to make the app feel a bit nicer
        if predicted_class == "no_tumor":
            st.write("Good news - no tumor was detected in this scan (according to the model).")
        else:
            st.write(f"The model thinks this scan shows signs of a **{predicted_class}** tumor.")

        # showing confidence scores for ALL classes, not just the top one
        st.subheader("Confidence for each class")
        results_df = pd.DataFrame({
            "Tumor Type": CLASS_NAMES,
            "Confidence (%)": [round(p * 100, 2) for p in predictions]
        }).sort_values(by="Confidence (%)", ascending=False)

        st.bar_chart(results_df.set_index("Tumor Type"))
        st.dataframe(results_df, use_container_width=True)

else:
    st.write("👆 Please upload an MRI image to get started.")


# ---------------------------------------------------------
st.markdown("---")
st.caption("Made as part of the GUVI-HCL Data Science capstone project.")
