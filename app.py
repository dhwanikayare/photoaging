import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("photoaging_model_v1.keras")

model = load_model()

IMG_SIZE = (224, 224)
BASE_MODEL_NAME = "mobilenetv2_1.00_224"
LAST_CONV_LAYER_NAME = "Conv_1"

# =============================
# LOAD AQI DATA
# =============================
@st.cache_data
def load_aqi():
    df = pd.read_csv("AQI and Lat Long of Countries.csv")

    df = df[["City", "Country", "PM2.5 AQI Value"]].dropna()
    df.columns = ["city", "country", "pm25"]
    df["city"] = df["city"].str.lower()

    return df

aqi_df = load_aqi()

def get_city_pm25(city):
    city = city.strip().lower()
    match = aqi_df[aqi_df["city"] == city]

    if len(match) > 0:
        return float(match.iloc[0]["pm25"])
    return None

def normalize_pm25(aqi):
    if aqi is None:
        return 0.6
    elif aqi <= 50:
        return 0.2
    elif aqi <= 100:
        return 0.4
    elif aqi <= 150:
        return 0.6
    elif aqi <= 200:
        return 0.8
    else:
        return 1.0

def clamp01(x):
    return float(max(0.0, min(1.0, x)))

def category(score):
    if score < 0.33:
        return "Low"
    elif score < 0.66:
        return "Moderate"
    return "High"

# =============================
# GRAD-CAM
# =============================
def make_gradcam(img_array):
    x = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    base_model = model.get_layer(BASE_MODEL_NAME)

    feature_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer(LAST_CONV_LAYER_NAME).output
    )

    classifier = model.get_layer(index=-1)

    with tf.GradientTape() as tape:
        conv_out = feature_extractor(x)
        tape.watch(conv_out)

        gap = tf.keras.layers.GlobalAveragePooling2D()(conv_out)
        logits = tf.matmul(gap, classifier.kernel) + classifier.bias
        loss = logits[:, 0]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

def overlay(img, heatmap):
    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)

    jet = plt.get_cmap("jet")
    colors = jet(np.arange(256))[:, :3]
    jet_heatmap = colors[heatmap]
    jet_heatmap = np.uint8(jet_heatmap * 255)

    return np.uint8(jet_heatmap * 0.4 + img)

# =============================
# UI
# =============================
st.title("Photoaging Burden Index")

st.write("Upload a face image and answer lifestyle questions")

image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

hours = st.slider("Hours outdoors per day", 0.0, 8.0, 2.0)
cigs = st.number_input("Cigarettes per day", 0, 40, 0)
city = st.text_input("City")
sunscreen = st.selectbox("Sunscreen daily?", ["yes", "no"])

if image and city:

    img = Image.open(image).convert("RGB").resize(IMG_SIZE)
    img_np = np.array(img).astype(np.float32)
    img_batch = np.expand_dims(img_np, axis=0)

    # CNN prediction
    x = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)
    pred = model.predict(x, verbose=0)
    P_vis = float(pred[0,0])

    # Questionnaire
    U = clamp01(hours / 6.0)
    S = clamp01(cigs / 20.0)

    pm25 = get_city_pm25(city)
    P = normalize_pm25(pm25)

    C = 1.0 if sunscreen == "yes" else 0.0

    # Weights
    R_exp = clamp01(
        0.70 * U +
        0.15 * S +
        0.10 * P +
        0.05 * (1 - C)
    )

    # Final score
    PBI = clamp01(0.80 * P_vis + 0.20 * R_exp)

    # GradCAM
    heatmap = make_gradcam(img_batch)
    cam = overlay(img_np.astype(np.uint8), heatmap)

    # Output
    st.subheader("Results")
    st.write("Visible Photoaging:", round(P_vis,4))
    st.write("Exposure Risk:", round(R_exp,4))
    st.write("Final Score:", round(PBI,4))
    st.write("Category:", category(PBI))

    if pm25:
        st.write(f"PM2.5 AQI: {pm25}")

    st.image(img_np.astype(np.uint8), caption="Original")
    st.image(cam, caption="Grad-CAM")