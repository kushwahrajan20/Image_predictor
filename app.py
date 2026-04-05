import streamlit as st
import tensorflow as tf
import keras
import numpy as np
from PIL import Image
import io
import requests

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔍",
    layout="centered",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background: #0d0d0d;
    color: #f0f0f0;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
}

.title-block {
    text-align: center;
    padding: 2rem 0 1rem 0;
}

.title-block h1 {
    font-size: 2.8rem;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #00f5a0, #00d9f5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.title-block p {
    color: #888;
    font-size: 1rem;
    font-family: 'Space Mono', monospace;
}

.result-card {
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-top: 1.5rem;
    text-align: center;
}

.result-real {
    background: linear-gradient(135deg, #0a2e1f, #0d3d28);
    border: 1px solid #00f5a0;
}

.result-fake {
    background: linear-gradient(135deg, #2e0a0a, #3d0d0d);
    border: 1px solid #f5003d;
}

.result-label {
    font-size: 2rem;
    font-weight: 800;
    font-family: 'Syne', sans-serif;
}

.result-score {
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    color: #aaa;
    margin-top: 0.4rem;
}

.confidence-bar-wrap {
    background: #1a1a1a;
    border-radius: 99px;
    height: 10px;
    margin-top: 1rem;
    overflow: hidden;
}

.confidence-bar {
    height: 10px;
    border-radius: 99px;
    transition: width 0.6s ease;
}

.stButton > button {
    background: linear-gradient(90deg, #00f5a0, #00d9f5);
    color: #000;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-size: 1rem;
    width: 100%;
    cursor: pointer;
}

.stButton > button:hover {
    opacity: 0.85;
}

.upload-hint {
    color: #555;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    text-align: center;
    margin-top: 0.5rem;
}

div[data-testid="stFileUploader"] {
    background: #111;
    border: 1px dashed #333;
    border-radius: 10px;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
MODEL_PATH = "./model/dualStreamModel2_ep9.keras"   # ← adjust path as needed

# ── FFT layer (must match training) ──────────────────────────────────────────
@keras.saving.register_keras_serializable()
def fft_layer(image):
    gray = tf.image.rgb_to_grayscale(image)
    gray = tf.squeeze(gray, axis=-1)
    fft = tf.signal.fft2d(tf.cast(gray, tf.complex64))
    fft_shift = tf.signal.fftshift(fft)
    magnitude = tf.math.log(tf.math.abs(fft_shift) + 1.0)
    mag_min = tf.reduce_min(magnitude, axis=[1, 2], keepdims=True)
    mag_max = tf.reduce_max(magnitude, axis=[1, 2], keepdims=True)
    magnitude = (magnitude - mag_min) / (mag_max - mag_min + 1e-7)
    magnitude = tf.expand_dims(magnitude, axis=-1)
    return magnitude

# ── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_pil(pil_image: Image.Image) -> tf.Tensor:
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="PNG")
    img_bytes = buf.getvalue()
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def preprocess_bytes(image_bytes: bytes) -> tf.Tensor:
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def predict(model, tensor: tf.Tensor):
    batch = tf.expand_dims(tensor, axis=0)
    score = model.predict(batch, verbose=0)[0][0]
    label = "AI-Generated" if score > 0.5 else "Real"
    confidence = float(score) if score > 0.5 else float(1 - score)
    return label, float(score), confidence

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>🔍 AI Image Detector</h1>
    <p>Dual-stream deep learning · FFT frequency analysis</p>
</div>
""", unsafe_allow_html=True)

# Load model
with st.spinner("Loading model…"):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

st.markdown("---")

# Input method
tab_upload, tab_url = st.tabs(["📁 Upload Image", "🔗 Image URL"])

tensor = None
display_img = None

with tab_upload:
    uploaded = st.file_uploader(
        "Drop an image here", type=["jpg", "jpeg", "png", "webp", "bmp"]
    )
    st.markdown('<p class="upload-hint">Supported: JPG · PNG · WEBP · BMP</p>',
                unsafe_allow_html=True)
    if uploaded:
        display_img = Image.open(uploaded)
        tensor = preprocess_pil(display_img)

with tab_url:
    url = st.text_input("Paste image URL", placeholder="https://example.com/image.jpg")
    if url:
        try:
            with st.spinner("Fetching image…"):
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                img_bytes = resp.content
                display_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                tensor = preprocess_bytes(img_bytes)
        except Exception as e:
            st.error(f"Could not fetch image: {e}")

# ── Predict ───────────────────────────────────────────────────────────────────
if tensor is not None and display_img is not None:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(display_img, use_container_width=True, caption="Input image")

    with col2:
        with st.spinner("Analyzing…"):
            label, score, confidence = predict(model, tensor)

        is_fake = label == "AI-Generated"
        card_class = "result-fake" if is_fake else "result-real"
        emoji = "🤖" if is_fake else "✅"
        color = "#f5003d" if is_fake else "#00f5a0"
        bar_pct = int(confidence * 100)

        st.markdown(f"""
        <div class="result-card {card_class}">
            <div class="result-label" style="color:{color};">{emoji} {label}</div>
            <div class="result-score">
                Raw score: {score:.4f} &nbsp;|&nbsp; Confidence: {bar_pct}%
            </div>
            <div class="confidence-bar-wrap">
                <div class="confidence-bar"
                     style="width:{bar_pct}%; background:{color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("####")
        st.markdown(f"""
        **How to read this:**  
        - Score **> 0.5** → model thinks it's AI-generated  
        - Score **< 0.5** → model thinks it's real  
        - Current score: `{score:.4f}`
        """)