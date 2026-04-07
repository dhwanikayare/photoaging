import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import difflib

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Photoaging Check",
    page_icon="🌿",
    layout="centered"
)

# ============================================================
# CUSTOM STYLING
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #fffaf5 0%, #f9f4ee 100%);
    }

    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero-box {
        background: linear-gradient(135deg, #f7e9dd 0%, #efe7fb 100%);
        border-radius: 24px;
        padding: 2rem 1.5rem;
        text-align: center;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        color: #4f3f34;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #6f6258;
        max-width: 700px;
        margin: 0 auto;
        line-height: 1.6;
    }

    .section-card {
        background: #ffffff;
        border-radius: 20px;
        padding: 1.2rem 1.2rem 1rem 1.2rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.06);
    }

    .small-note {
        font-size: 0.92rem;
        color: #7d6f65;
        line-height: 1.5;
    }

    .result-box {
        border-radius: 18px;
        padding: 1rem 1.1rem;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        font-size: 1rem;
        line-height: 1.6;
    }

    .result-low {
        background-color: #edf8ef;
        border-left: 6px solid #63b174;
        color: #245d31;
    }

    .result-moderate {
        background-color: #fff7e9;
        border-left: 6px solid #e6b24d;
        color: #7b5811;
    }

    .result-high {
        background-color: #fdeeee;
        border-left: 6px solid #d96a6a;
        color: #7c2525;
    }

    .fact-box {
        background: #f7f4ff;
        border-radius: 16px;
        padding: 0.9rem 1rem;
        border: 1px solid #ebe3ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("photoaging_model_v1.keras")

model = load_model()
IMG_SIZE = (224, 224)

# ============================================================
# LOAD AQI DATA
# Supports either:
# 1. City, Country, PM2.5 AQI Value
# 2. city, country, pm25_aqi
# ============================================================
@st.cache_data
def load_aqi():
    df = pd.read_csv("city_pm25_aqi.csv")

    original_cols = set(df.columns)

    if {"City", "Country", "PM2.5 AQI Value"}.issubset(original_cols):
        df = df[["City", "Country", "PM2.5 AQI Value"]].dropna().copy()
        df.columns = ["city", "country", "pm25_aqi"]

    elif {"city", "country", "pm25_aqi"}.issubset(original_cols):
        df = df[["city", "country", "pm25_aqi"]].dropna().copy()

    else:
        raise ValueError(
            "The AQI CSV must contain either "
            "['City', 'Country', 'PM2.5 AQI Value'] "
            "or ['city', 'country', 'pm25_aqi'] columns."
        )

    df["city"] = df["city"].astype(str).str.strip().str.lower()
    df["country"] = df["country"].astype(str).str.strip().str.lower()
    df = df.groupby(["city", "country"], as_index=False)["pm25_aqi"].mean()

    return df

aqi_df = load_aqi()

# ============================================================
# HELPERS
# ============================================================
def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def normalize_pm25(aqi):
    if aqi is None:
        return 0.6
    if aqi <= 50:
        return 0.2
    elif aqi <= 100:
        return 0.4
    elif aqi <= 150:
        return 0.6
    elif aqi <= 200:
        return 0.8
    return 1.0

def category(score):
    if score < 0.33:
        return "Low"
    elif score < 0.66:
        return "Moderate"
    return "High"

def get_city_pm25(city_input):
    city_input = city_input.strip().lower()
    matches = aqi_df[aqi_df["city"] == city_input]

    if len(matches) == 0:
        return None, None, 0

    if len(matches) == 1:
        row = matches.iloc[0]
        return float(row["pm25_aqi"]), row["country"], 1

    mean_pm25 = float(matches["pm25_aqi"].mean())
    countries = ", ".join(sorted(matches["country"].dropna().astype(str).str.title().unique()))
    return mean_pm25, countries, len(matches)

def get_city_suggestions(city_input, n=5):
    city_input = city_input.strip().lower()
    city_list = sorted(aqi_df["city"].dropna().unique().tolist())
    suggestions = difflib.get_close_matches(city_input, city_list, n=n, cutoff=0.6)
    return [s.title() for s in suggestions]

def get_risk_text(risk_label):
    if risk_label == "Low":
        return "Your image currently shows limited visible signs associated with photoaging."
    elif risk_label == "Moderate":
        return "Your image shows some visible signs associated with photoaging, and your lifestyle profile suggests moderate cumulative exposure."
    return "Your image shows stronger visible signs associated with photoaging, and your lifestyle profile suggests elevated cumulative exposure."

def build_recommendations(hours, cigs, pollution_score, sunscreen):
    tips = []

    if sunscreen == "no":
        tips.append("Try using sunscreen daily, especially before prolonged outdoor exposure.")

    if hours >= 4:
        tips.append("Consider reducing long periods of direct sun exposure, especially around midday.")

    if cigs > 0:
        tips.append("Smoking is associated with faster visible skin aging. Reducing or stopping can support skin health.")

    if pollution_score >= 0.6:
        tips.append("Higher pollution exposure may contribute to skin stress. Cleansing and supportive skincare habits may help.")

    if hours <= 2 and cigs == 0 and sunscreen == "yes" and pollution_score < 0.6:
        tips.append("Your current habits look relatively protective. Maintaining them may help reduce long-term photoaging risk.")

    return tips

def get_skin_routine(risk_label):
    if risk_label == "Low":
        return [
            "Use a gentle cleanser morning and evening.",
            "Apply sunscreen daily with broad-spectrum SPF 30 or higher.",
            "Use a simple moisturizer to support skin barrier health.",
            "Continue maintaining protective habits such as limiting unnecessary sun exposure."
        ]
    elif risk_label == "Moderate":
        return [
            "Cleanse gently twice daily and avoid harsh scrubbing.",
            "Apply broad-spectrum SPF 30 or higher every morning and reapply if outdoors for long periods.",
            "Use a moisturizer consistently to support skin barrier health.",
            "Consider adding antioxidant-based skincare to support protection against environmental stress.",
            "Reduce prolonged direct sun exposure where possible."
        ]
    else:
        return [
            "Use a gentle, non-irritating cleanser morning and evening.",
            "Apply broad-spectrum SPF 50 every morning and reapply regularly when outdoors.",
            "Use a barrier-supporting moisturizer daily.",
            "Consider antioxidant and pigment-supportive skincare if appropriate.",
            "Minimize prolonged sun exposure and seek shade during peak UV hours.",
            "If you are concerned about visible skin changes, consider consulting a dermatologist."
        ]

def predict_visible_photoaging(img_pil):
    img = img_pil.convert("RGB").resize(IMG_SIZE)
    img_np = np.array(img).astype(np.float32)
    img_batch = np.expand_dims(img_np, axis=0)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)
    pred = model.predict(x, verbose=0)
    return float(pred[0, 0]), img

# ============================================================
# HERO SECTION
# ============================================================
st.markdown(
    """
    <div class='hero-box'>
        <div class='hero-title'>Photoaging Check</div>
        <div class='hero-subtitle'>
            Upload a face image or take a picture, answer a few lifestyle questions,
            and get a simple estimate of your visible photoaging risk with practical skin safety suggestions.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# IMAGE INPUT
# ============================================================
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("1. Add your face image")
st.markdown(
    "<div class='small-note'>For best results, use a clear frontal face image with minimal background and good lighting.</div>",
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["Upload Image", "Take a Picture"])
selected_image = None

with tab1:
    uploaded_image = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        selected_image = uploaded_image

with tab2:
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        selected_image = camera_image

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# QUESTIONNAIRE
# ============================================================
with st.form("photoaging_form"):
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("2. Lifestyle questionnaire")

    hours = st.slider("How many hours do you usually spend outdoors during daylight each day?", 0.0, 8.0, 2.0, 0.5)
    cigs = st.number_input("How many cigarettes do you smoke per day?", min_value=0, max_value=40, value=0, step=1)
    city = st.text_input("Which city do you live in most of the time?")
    sunscreen = st.selectbox("Do you apply sunscreen daily?", ["yes", "no"])

    submitted = st.form_submit_button("Check My Photoaging Risk")
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# RUN
# ============================================================
if submitted:
    if selected_image is None:
        st.error("Please upload an image or take a picture before submitting.")
    elif not city.strip():
        st.error("Please enter your city.")
    else:
        img_pil = Image.open(selected_image)
        visible_score, display_img = predict_visible_photoaging(img_pil)

        uv_score = clamp01(hours / 6.0)
        smoking_score = clamp01(cigs / 20.0)

        pm25_value, matched_country, city_matches = get_city_pm25(city)
        pollution_score = normalize_pm25(pm25_value)

        sunscreen_protection = 1.0 if sunscreen == "yes" else 0.0

        exposure_score = clamp01(
            0.70 * uv_score +
            0.15 * smoking_score +
            0.10 * pollution_score +
            0.05 * (1 - sunscreen_protection)
        )

        final_score = clamp01(0.80 * visible_score + 0.20 * exposure_score)
        risk_label = category(final_score)
        risk_text = get_risk_text(risk_label)
        tips = build_recommendations(hours, cigs, pollution_score, sunscreen)
        routine = get_skin_routine(risk_label)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("3. Your result")

        if risk_label == "Low":
            st.markdown(f"<div class='result-box result-low'><b>Low Photoaging Risk</b><br>{risk_text}</div>", unsafe_allow_html=True)
        elif risk_label == "Moderate":
            st.markdown(f"<div class='result-box result-moderate'><b>Moderate Photoaging Risk</b><br>{risk_text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-box result-high'><b>High Photoaging Risk</b><br>{risk_text}</div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Visible Skin Change", f"{visible_score:.2f}")
        col2.metric("Exposure Profile", f"{exposure_score:.2f}")
        col3.metric("Overall Result", risk_label)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("What this means")
        st.write(
            "Photoaging refers to visible skin changes linked to long-term environmental exposure, especially ultraviolet radiation. "
            "It may appear as wrinkles, dark spots, uneven skin tone, or reduced skin elasticity."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Personalized suggestions")
        for tip in tips:
            st.write(f"• {tip}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Suggested skin care routine")
        for step in routine:
            st.write(f"• {step}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Helpful skin health facts")
        st.markdown("<div class='fact-box'>", unsafe_allow_html=True)
        st.write("• Chronic sun exposure is the main external factor linked to photoaging.")
        st.write("• Visible photoaging can appear as wrinkles, dark spots, uneven tone, and texture changes.")
        st.write("• Consistent sunscreen use can help reduce cumulative UV-related skin damage.")
        st.write("• Smoking and environmental pollution may contribute to long-term skin stress.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Pollution information")
        if pm25_value is not None:
            if city_matches == 1:
                st.write(f"Matched city data: **{city.strip().title()}**, **{matched_country.title()}**")
            else:
                st.write(f"Matched city data from **{city_matches} records** for **{city.strip().title()}** across: **{matched_country}**")
            st.write(f"Estimated PM2.5 AQI: **{pm25_value:.1f}**")
        else:
            suggestions = get_city_suggestions(city)
            st.warning("City pollution data was not found. A default pollution estimate was used.")
            if suggestions:
                st.write("Did you mean:")
                for s in suggestions:
                    st.write(f"• {s}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Uploaded image")
        st.image(display_img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.caption(
            "This tool is intended for educational and research purposes only. "
            "It does not provide a medical diagnosis or replace professional dermatological advice."
        )
