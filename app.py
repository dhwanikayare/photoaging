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

    # Rename original CSV columns to clean internal names
    df = df[["City", "Country", "PM2.5 AQI Value"]].dropna()
    df.columns = ["city", "country", "pm25_aqi"]

    df["city"] = df["city"].str.lower().str.strip()
    df["country"] = df["country"].str.lower().str.strip()

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
# UI
# =============================
st.set_page_config(page_title="Photoaging Burden Index", layout="centered")

st.title("Photoaging Burden Index")
st.write(
    "Upload a clear frontal face image or take one using your camera, then answer a few lifestyle questions."
)

# -----------------------------
# IMAGE INPUT
# -----------------------------
st.subheader("Image Input")
tab1, tab2 = st.tabs(["Upload Image", "Take a Picture"])

uploaded_image = None

with tab1:
    uploaded_image = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

with tab2:
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        uploaded_image = camera_image

# -----------------------------
# FORM (WITH SUBMIT BUTTON)
# -----------------------------
with st.form("photoaging_form"):

    st.subheader("Lifestyle Questionnaire")

    hours = st.slider("Average hours outdoors per day", 0.0, 8.0, 2.0, 0.5)
    cigs = st.number_input("Cigarettes per day", 0, 40, 0)
    city = st.text_input("City")
    sunscreen = st.selectbox("Daily sunscreen use", ["yes", "no"])

    submitted = st.form_submit_button("Calculate Photoaging Score")

# -----------------------------
# RUN ONLY AFTER SUBMIT
# -----------------------------
if submitted:

    if uploaded_image is None:
        st.error("Please upload or capture an image.")
    elif not city:
        st.error("Please enter your city.")
    else:
        # Image processing
        img = Image.open(uploaded_image).convert("RGB").resize(IMG_SIZE)
        img_np = np.array(img).astype(np.float32)
        img_batch = np.expand_dims(img_np, axis=0)

        # CNN prediction
        x = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)
        pred = model.predict(x, verbose=0)
        P_vis = float(pred[0, 0])

        # Questionnaire
        U = clamp01(hours / 6.0)
        S = clamp01(cigs / 20.0)

        pm25 = get_city_pm25(city)
        P = normalize_pm25(pm25)

        C = 1.0 if sunscreen == "yes" else 0.0

        # Exposure risk
        R_exp = clamp01(
            0.70 * U +
            0.15 * S +
            0.10 * P +
            0.05 * (1 - C)
        )

        # Final score
        PBI = clamp01(0.80 * P_vis + 0.20 * R_exp)

        # -----------------------------
        # RESULTS
        # -----------------------------
        st.subheader("Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("Facial Score", f"{P_vis:.3f}")
        col2.metric("Exposure Risk", f"{R_exp:.3f}")
        col3.metric("Final PBI", f"{PBI:.3f}")

        st.write("Severity Category:", category(PBI))

        if pm25 is not None:
            st.write(f"PM2.5 AQI for {city.title()}: {pm25:.1f}")
        else:
            st.warning("City not found. Default pollution score used.")

        st.image(img, caption="Input Face Image", use_container_width=True)
