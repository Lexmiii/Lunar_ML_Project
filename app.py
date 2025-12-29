import streamlit as st
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
}
.footer {
    text-align: center;
    font-size: 12px;
    color: #6e7681;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown("<div class='title'>Lunar Surface Classification</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>CNN-based Machine Learning Model</div>", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Lunar_cnn_model.keras")

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
    st.image(image, use_column_width=True)

    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    st.markdown("<hr>", unsafe_allow_html=True)

    if prediction > 0.5:
        st.markdown(
            f"<div class='result'>Crater Surface Detected<br>Confidence: {prediction:.2f}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result'>Smooth Surface Detected<br>Confidence: {1 - prediction:.2f}</div>",
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown(
    "<div class='footer'>Built with TensorFlow & Streamlit | Lunar ML Project</div>",
    unsafe_allow_html=True
)