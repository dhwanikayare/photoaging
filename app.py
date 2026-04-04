import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("photoaging_model_v1.keras")

model = load_model()

IMG_SIZE = (224, 224)

# =============================
# LOAD AQI DATA
# =============================
@st.cache_data
def load_aqi():
    df = pd.read_csv("city_pm25_aqi.csv")
    df = df[["city", "country", "pm25_aqi"]].dropna()
    df["city"] = df["city"].str.lower().str.strip()
    return df

aqi_df = load_aqi()

def get_city_pm25(city):
    city = city.strip().lower()
    match = aqi_df[aqi_df["city"] == city]
    if len(match) > 0:
        return float(match.iloc[0]["pm25_aqi"])
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
# PAGE UI
# =============================
st.set_page_config(page_title="Photoaging Burden Index", layout="centered")

st.title("Photoaging Burden Index")
st.write(
    "Upload a clear frontal face image or take one using your camera, then answer a few lifestyle questions."
)

st.subheader("Image Input")
tab1, tab2 = st.tabs(["Upload Image", "Take a Picture"])

uploaded_image = None

with tab1:
    uploaded_image = st.file_uploader(
        "Upload a face image",
        type=["jpg", "jpeg", "png"]
    )

with tab2:
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        uploaded_image = camera_image

st.subheader("Lifestyle Questionnaire")
hours = st.slider("Average hours spent outdoors per day", 0.0, 8.0, 2.0, 0.5)
cigs = st.number_input("Cigarettes per day", min_value=0, max_value=40, value=0, step=1)
city = st.text_input("Which city do you live in most of the time?")
sunscreen = st.selectbox("Do you apply sunscreen daily?", ["yes", "no"])

if uploaded_image is not None and city:

    img = Image.open(uploaded_image).convert("RGB").resize(IMG_SIZE)
    img_np = np.array(img).astype(np.float32)
    img_batch = np.expand_dims(img_np, axis=0)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)
    pred = model.predict(x, verbose=0)
    P_vis = float(pred[0, 0])

    U = clamp01(hours / 6.0)
    S = clamp01(cigs / 20.0)
    pm25 = get_city_pm25(city)
    P = normalize_pm25(pm25)
    C = 1.0 if sunscreen == "yes" else 0.0

    R_exp = clamp01(
        0.70 * U +
        0.15 * S +
        0.10 * P +
        0.05 * (1 - C)
    )

    PBI = clamp01(0.80 * P_vis + 0.20 * R_exp)

    st.subheader("Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Visible Photoaging", f"{P_vis:.3f}")
    c2.metric("Exposure Risk", f"{R_exp:.3f}")
    c3.metric("Final PBI", f"{PBI:.3f}")

    st.write("Severity Category:", category(PBI))

    if pm25 is not None:
        st.write(f"PM2.5 AQI for {city.title()}: {pm25:.1f}")
    else:
        st.write("City pollution data not found. A default pollution score was used.")

    st.image(img, caption="Input Face Image", use_container_width=True)
