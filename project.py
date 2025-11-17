import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Flat Foot Detection", layout="wide")

st.title("🦶 Flat Foot Detection using EfficientNet + Autoencoder + GradCAM")
st.write("Upload your foot image to classify Flat Foot vs Normal Foot.")

# -----------------------------------------------------------------------------------
# 1. AUTOENCODER MODEL (Feature Learning)
# -----------------------------------------------------------------------------------

def build_autoencoder():
    input_img = Input(shape=(224, 224, 3))
    
    # Encoder
    x = Conv2D(32, 3, activation='relu', padding='same')(input_img)
    x = MaxPooling2D(2, padding='same')(x)
    x = Conv2D(16, 3, activation='relu', padding='same')(x)
    encoded = MaxPooling2D(2, padding='same')(x)

    # Decoder
    x = Conv2D(16, 3, activation='relu', padding='same')(encoded)
    x = UpSampling2D()(x)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = UpSampling2D()(x)
    
    decoded = Conv2D(3, 3, activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder


# -----------------------------------------------------------------------------------
# 2. CLASSIFICATION MODEL (EfficientNetB0)
# -----------------------------------------------------------------------------------

def build_classifier():
    base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224,224,3))
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Load or build models
autoencoder = build_autoencoder()
classifier = build_classifier()

# -----------------------------------------------------------------------------------
# 3. GRAD-CAM FUNCTION
# -----------------------------------------------------------------------------------

def generate_gradcam(model, img_array, last_conv_layer="top_conv"):
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer).output, model.output])

    with tf.GradientTape() as tape:
        conv_output, prediction = grad_model(img_array)
        loss = prediction[:, 0]

    grads = tape.gradient(loss, conv_output)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0,1))

    conv_output = conv_output[0]

    for i in range(pooled_grads.shape[-1]):
        conv_output[:,:,i] *= pooled_grads[i]

    heatmap = np.mean(conv_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    
    if heatmap.max() == 0:
        heatmap = heatmap
    else:
        heatmap /= heatmap.max()

    heatmap = cv2.resize(heatmap, (224,224))
    return heatmap


# -----------------------------------------------------------------------------------
# 4. STREAMLIT UI + PREDICTION
# -----------------------------------------------------------------------------------

uploaded_file = st.file_uploader("Upload Foot Image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    img_resized = image.resize((224,224))
    img_array = img_to_array(img_resized) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_column_width=True)

    # AUTOENCODER RECONSTRUCTION
    reconstructed = autoencoder.predict(img_input)[0]

    with col2:
        st.subheader("Autoencoder Reconstruction")
        st.image(reconstructed, use_column_width=True)

    # PREDICT FLAT FOOT / NORMAL
    pred = classifier.predict(img_input)[0][0]
    label = "🟥 FLAT FOOT DETECTED" if pred > 0.5 else "🟩 NORMAL FOOT"
    prob = pred if pred > 0.5 else 1 - pred

    st.markdown(f"## **Prediction: {label}**")
    st.markdown(f"### Confidence: **{prob*100:.2f}%**")

    # GRAD-CAM HEATMAP
    heatmap = generate_gradcam(classifier, img_input)
    
    # Overlay heatmap on image
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    original = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
    superimposed_img = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

    st.subheader("Grad-CAM Explainability")
    st.image(superimposed_img, channels="BGR", use_column_width=True)


st.markdown("---")
st.write("✔ EfficientNetB0  | ✔ Autoencoder Features | ✔ Grad-CAM Explainability")
st.write("Built with ❤️ using Streamlit + TensorFlow")
