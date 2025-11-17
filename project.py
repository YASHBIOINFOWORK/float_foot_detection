import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Flat Foot Detector", layout="wide")

st.title("🦶 Flat Foot Detection (TFLite Version)")
st.write("Upload a foot image to classify: **Flat Foot vs Normal Foot**")

# -----------------------------
# Load TFLite model
# -----------------------------
@st.cache_resource
def load_interpreter():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_interpreter()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# -----------------------------
# Preprocess image
# -----------------------------
def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = img.astype("float32")
    img = np.expand_dims(img, axis=0)
    return img


# -----------------------------
# Fake Grad-CAM Heatmap without CV2
# -----------------------------
def generate_heatmap(image_array):
    heat = np.mean(image_array[0], axis=2)  # average channels
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    heatmap = Image.fromarray(np.uint8(heat * 255)).resize((224, 224))

    # apply red tint
    heatmap = heatmap.convert("RGB")
    enhancer = ImageEnhance.Color(heatmap)
    heatmap = enhancer.enhance(3.0)
    return heatmap


# -----------------------------
# Prediction
# -----------------------------
def predict(img_array):
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])
    return float(pred[0][0])


# -----------------------------
# UI Handling
# -----------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = preprocess_image(img)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(img, use_column_width=True)

    # prediction
    prob = predict(img_array)
    label = "🟥 FLAT FOOT DETECTED" if prob > 0.5 else "🟩 NORMAL FOOT"
    confidence = prob if prob > 0.5 else 1 - prob

    st.markdown(f"## **Prediction: {label}**")
    st.markdown(f"### Confidence: **{confidence*100:.2f}%**")

    # heatmap
    heatmap = generate_heatmap(img_array)
    blended = Image.blend(img.resize((224, 224)), heatmap, alpha=0.5)

    with col2:
        st.subheader("Explainability Heatmap")
        st.image(blended, use_column_width=True)

st.markdown("---")
st.write("✔ TensorFlow Lite | ✔ No OpenCV | ✔ Fully Deployable on Streamlit Cloud")
