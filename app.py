import streamlit as st
import tensorflow as tf
import keras
import numpy as np
from PIL import Image
import io
import requests

st.set_page_config(
    page_title="LENS — AI Image Detector",
    page_icon="◉",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #080808 !important;
    color: #e8e8e2;
}

.stApp {
    background: #080808 !important;
}

/* Noise texture overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.5;
}

/* ── HEADER ── */
.lens-header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    padding: 3.5rem 0 2rem 0;
    border-bottom: 1px solid #1e1e1e;
    margin-bottom: 2.5rem;
}

.lens-wordmark {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5.5rem;
    line-height: 0.9;
    letter-spacing: 4px;
    color: #e8e8e2;
    position: relative;
}

.lens-wordmark span {
    color: #c8f135;
}

.lens-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #444;
    text-transform: uppercase;
    letter-spacing: 2px;
    text-align: right;
    line-height: 1.8;
}

/* ── TABS ── */
div[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1e1e1e !important;
    gap: 0 !important;
}

div[data-baseweb="tab"] {
    background: transparent !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    color: #444 !important;
    border: none !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 0 !important;
}

div[data-baseweb="tab"][aria-selected="true"] {
    color: #c8f135 !important;
    border-bottom: 2px solid #c8f135 !important;
}

div[data-baseweb="tab-panel"] {
    padding: 1.5rem 0 0 0 !important;
}

/* ── FILE UPLOADER ── */
div[data-testid="stFileUploader"] {
    background: #0e0e0e !important;
    border: 1px dashed #2a2a2a !important;
    border-radius: 4px !important;
    padding: 2rem !important;
    transition: border-color 0.2s;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #c8f135 !important;
}

div[data-testid="stFileUploader"] label {
    color: #555 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* ── TEXT INPUT ── */
div[data-baseweb="input"] {
    background: #0e0e0e !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 4px !important;
}

div[data-baseweb="input"]:focus-within {
    border-color: #c8f135 !important;
    box-shadow: 0 0 0 1px #c8f135 !important;
}

input {
    font-family: 'DM Mono', monospace !important;
    color: #e8e8e2 !important;
    background: transparent !important;
    font-size: 0.85rem !important;
}

input::placeholder {
    color: #333 !important;
}

/* ── BUTTON ── */
.stButton > button {
    background: #c8f135 !important;
    color: #080808 !important;
    font-family: 'DM Mono', monospace !important;
    font-weight: 500 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 0.75rem 2.5rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: opacity 0.15s, transform 0.1s !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
    transform: translateY(-1px) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── SPINNER ── */
.stSpinner > div {
    border-color: #c8f135 transparent transparent transparent !important;
}

/* ── RESULT PANEL ── */
.result-wrap {
    background: #0e0e0e;
    border: 1px solid #1e1e1e;
    border-radius: 4px;
    padding: 2rem;
    display: flex;
    flex-direction: column;
    gap: 1.2rem;
    height: 100%;
}

.result-status-line {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

.result-verdict {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.8rem;
    line-height: 1;
    letter-spacing: 2px;
}

.result-divider {
    height: 1px;
    background: #1e1e1e;
}

.result-metric-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
}

.result-metric {
    background: #141414;
    border-radius: 3px;
    padding: 0.9rem 1rem;
}

.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #444;
    margin-bottom: 0.3rem;
}

.metric-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.4rem;
    font-weight: 500;
}

.bar-track {
    height: 3px;
    background: #1a1a1a;
    border-radius: 2px;
    overflow: hidden;
}

.bar-fill {
    height: 3px;
    border-radius: 2px;
}

.result-note {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #333;
    line-height: 1.7;
    border-left: 2px solid #1e1e1e;
    padding-left: 0.75rem;
}

/* ── IMAGE DISPLAY ── */
.stImage img {
    border-radius: 4px !important;
    border: 1px solid #1e1e1e !important;
}

/* ── FOOTER ── */
.lens-footer {
    margin-top: 4rem;
    padding: 2rem 0;
    border-top: 1px solid #141414;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.footer-left {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #2a2a2a;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.footer-right {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #2a2a2a;
}

/* ── ALERTS ── */
div[data-baseweb="notification"] {
    background: #120e00 !important;
    border: 1px solid #3a2e00 !important;
    border-radius: 4px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ── COLUMNS ── */
[data-testid="column"] {
    padding: 0 !important;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──
IMG_SIZE = (224, 224)
MODEL_PATH = "./model/dualStreamModel2_ep10.keras"

# ── FFT layer ──
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

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

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

# ── HEADER ──
st.markdown("""
<div class="lens-header">
    <div class="lens-wordmark">LEN<span>S</span></div>
    <div class="lens-sub">
        AI Image Detector<br>
        Dual-Stream · FFT Analysis<br>
        v2.0 — Deep Learning
    </div>
</div>
""", unsafe_allow_html=True)

# ── Model ──
with st.spinner("Initialising model…"):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()

# ── Input Tabs ──
tab_upload, tab_url = st.tabs(["Upload Image", "Image URL"])

tensor = None
display_img = None

with tab_upload:
    uploaded = st.file_uploader(
        "Drop an image here — JPG · PNG · WEBP · BMP",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="visible"
    )
    if uploaded:
        display_img = Image.open(uploaded)
        tensor = preprocess_pil(display_img)

with tab_url:
    url = st.text_input(
        "Image URL",
        placeholder="https://example.com/image.jpg",
        label_visibility="collapsed"
    )
    if url:
        try:
            with st.spinner("Fetching…"):
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                img_bytes = resp.content
                display_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                tensor = preprocess_bytes(img_bytes)
        except Exception as e:
            st.error(f"Could not fetch image: {e}")

# ── Predict + Display ──
if tensor is not None and display_img is not None:
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(display_img, use_container_width=True)

    with col2:
        with st.spinner("Analysing…"):
            label, score, confidence = predict(model, tensor)

        is_fake = label == "AI-Generated"
        accent   = "#c8f135" if not is_fake else "#ff4d4d"
        verdict  = "AI-GENERATED" if is_fake else "AUTHENTIC"
        bar_pct  = int(confidence * 100)
        score_display = f"{score:.4f}"
        conf_display  = f"{bar_pct}%"
        signal = "SYNTHETIC SIGNAL DETECTED" if is_fake else "NO SYNTHETIC SIGNAL"

        st.markdown(f"""
        <div class="result-wrap">
            <div class="result-status-line">
                <div class="status-dot" style="background:{accent};"></div>
                <span style="font-family:'DM Mono',monospace;font-size:0.65rem;
                             text-transform:uppercase;letter-spacing:1.5px;color:#444;">
                    {signal}
                </span>
            </div>

            <div class="result-verdict" style="color:{accent};">{verdict}</div>

            <div class="result-divider"></div>

            <div class="result-metric-row">
                <div class="result-metric">
                    <div class="metric-label">Raw Score</div>
                    <div class="metric-value" style="color:{accent};">{score_display}</div>
                </div>
                <div class="result-metric">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value" style="color:{accent};">{conf_display}</div>
                </div>
            </div>

            <div>
                <div class="bar-track">
                    <div class="bar-fill"
                         style="width:{bar_pct}%;background:{accent};"></div>
                </div>
            </div>

            <div class="result-note">
                Score &gt; 0.5 → AI-generated<br>
                Score &lt; 0.5 → Authentic<br>
                Current: {score_display}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("""
<div class="lens-footer">
    <div class="footer-left">LENS · AI Detection System</div>
    <div class="footer-right">Dual-Stream CNN + FFT</div>
</div>
""", unsafe_allow_html=True)