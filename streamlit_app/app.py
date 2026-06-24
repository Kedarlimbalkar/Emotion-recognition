"""
Streamlit Web App - DeepFER Emotion Recognition
Model: Custom CNN, Input 48x48 grayscale, 7 emotions
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import plotly.graph_objects as go
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import EMOTIONS, IMG_SIZE, MODEL_PATH

st.set_page_config(page_title="DeepFER - Emotion Recognition", page_icon="🎭", layout="wide")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found at `{MODEL_PATH}`.\nPlease run `python fix_model.py` first.")
        return None
    return tf.keras.models.load_model(MODEL_PATH)

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def preprocess(face_img):
    """48x48 grayscale — matches training."""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    normalized = resized.astype('float32') / 255.0
    return normalized.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)

def predict_emotion(model, face_img):
    processed = preprocess(face_img)
    preds = model.predict(processed, verbose=0)[0]
    idx = np.argmax(preds)
    return EMOTIONS[idx], float(preds[idx]), dict(zip(EMOTIONS, preds.tolist()))

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🎭 DeepFER")
st.sidebar.markdown("**Model:** Custom CNN")
st.sidebar.markdown("**Input:** 48×48 Grayscale")
st.sidebar.markdown("**Emotions:** 7 classes")
page = st.sidebar.radio("Navigate", ["📷 Image Detection", "📊 Model Info"])

model = load_model()

# ── Image Detection ───────────────────────────────────────────────────────────
if page == "📷 Image Detection":
    st.title("🎭 Facial Emotion Recognition")
    uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

    if uploaded and model:
        pil_img = Image.open(uploaded).convert("RGB")
        img_array = np.array(pil_img)
        frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input Image")
            st.image(pil_img, use_column_width=True)

        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        if len(faces) == 0:
            # No face — predict on full image
            st.warning("⚠️ No face detected — predicting on full image.")
            emotion, confidence, all_preds = predict_emotion(model, frame)
            faces_data = [{'emotion': emotion, 'confidence': confidence,
                           'all_predictions': all_preds, 'bbox': None}]
        else:
            faces_data = []
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                emotion, confidence, all_preds = predict_emotion(model, face_img)
                faces_data.append({'emotion': emotion, 'confidence': confidence,
                                   'all_predictions': all_preds, 'bbox': (x,y,w,h)})
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(frame, f"{emotion} {confidence*100:.0f}%",
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        with col2:
            st.subheader("Results")
            for r in faces_data:
                st.success(f"**Detected:** {r['emotion'].upper()}")
                st.metric("Confidence", f"{r['confidence']*100:.1f}%")

                fig = go.Figure(go.Bar(
                    x=list(r['all_predictions'].keys()),
                    y=[v*100 for v in r['all_predictions'].values()],
                    marker_color=['#FF4B4B' if k == r['emotion'] else '#636EFA'
                                  for k in r['all_predictions'].keys()]
                ))
                fig.update_layout(yaxis_title="Confidence (%)",
                                  xaxis_title="Emotion", height=300)
                st.plotly_chart(fig, use_column_width=True)

        if len(faces) > 0:
            annotated = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.subheader("Annotated Image")
            st.image(annotated, use_column_width=True)

# ── Model Info ────────────────────────────────────────────────────────────────
elif page == "📊 Model Info":
    st.title("📊 Model Information")
    st.markdown("""
    ## DeepFER - Custom CNN Architecture

    | Layer | Details |
    |-------|---------|
    | Input | 48×48×1 (Grayscale) |
    | Conv2D | 32 filters, 3×3, ReLU |
    | BatchNormalization | Stabilizes training |
    | MaxPooling2D | 2×2 |
    | Flatten | — |
    | Dense | 64 units, ReLU |
    | Output | 7 units, Softmax |

    ## Dataset
    - **Training:** 28,839 images
    - **Validation:** 7,066 images
    - **Classes:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise

    ## Files
    - `models/deepfer_compatible.h5` — Compatible model (run fix_model.py first)
    - `models/DeepFER_Final_Model_v1.keras` — Original trained model
    """)
    if model:
        st.success("✅ Model loaded and ready!")
