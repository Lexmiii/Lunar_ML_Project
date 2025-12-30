import streamlit as st
import os
import urllib.request

import numpy as np
from PIL import Image
import tensorflow as tf

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Lunar Surface Classifier",
    layout="centered"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}
.main {
    background-color: #0e1117;
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
}
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #9aa0a6;
    margin-bottom: 30px;
}
.card {
    background-color: #161b22;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 0px 20px rgba(0,0,0,0.4);
}
.result {
    font-size: 22px;
    font-weight: 600;
    text-align: center;
    margin-top: 10px;
}
.footer {
    text-align: center;
    font-size: 12px;
    color: #6e7681;
    margin-top: 40px;
}
hr {
    border: 0;
    height: 1px;
    background: #444;
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown("<div class='title'>Lunar Surface Classification</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>CNN-based Machine Learning Model</div>", unsafe_allow_html=True)

# ---------- MODEL DOWNLOAD ----------
MODEL_NAME = "lunar_cnn_model.keras"
MODEL_URL = "https://github.com/Lexmiii/Lunar_ML_Project/releases/download/v1.0/lunar_cnn_model.keras"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_NAME):
        with st.spinner("Downloading model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_NAME)
    return tf.keras.models.load_model(MODEL_NAME)

model = load_model()

# ---------- UI CARD ----------
st.markdown("<div class='card'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a lunar surface image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

IMG_SIZE = 64

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    # Preprocess
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        surface = "Crater Surface"
        landing_risk = "High Risk"
        confidence = prediction
    else:
        surface = "Smooth Surface"
        landing_risk = "Low Risk"
        confidence = 1 - prediction

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='result'><b>Surface:</b> {surface}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='result'><b>Confidence:</b> {confidence:.2f}</div>",
        unsafe_allow_html=True
    )

    risk_color = "red" if landing_risk == "High Risk" else "green"
    st.markdown(
        f"<div class='result' style='color:{risk_color}'><b>Landing Risk:</b> {landing_risk}</div>",
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown(
    "<div class='footer'>Built with TensorFlow & Streamlit | Lunar ML Project</div>",
    unsafe_allow_html=True
)