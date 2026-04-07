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
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CUSTOM STYLING
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --bg: #f7f3ee;
        --card: rgba(255, 255, 255, 0.78);
        --card-solid: #ffffff;
        --text: #1f2a37;
        --muted: #6b7280;
        --line: rgba(31, 41, 55, 0.08);
        --shadow: 0 10px 35px rgba(59, 38, 18, 0.08);
        --shadow-soft: 0 6px 20px rgba(59, 38, 18, 0.05);
        --accent: #d97757;
        --accent-2: #8b5cf6;
        --low-bg: #eef9f1;
        --low-border: #4aa366;
        --moderate-bg: #fff8e8;
        --moderate-border: #db9f2f;
        --high-bg: #fdeeee;
        --high-border: #d25d5d;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(234, 179, 122, 0.10), transparent 26%),
            radial-gradient(circle at top right, rgba(139, 92, 246, 0.10), transparent 28%),
            linear-gradient(180deg, #fcfaf8 0%, var(--bg) 100%);
        color: var(--text);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2.25rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: var(--text);
        letter-spacing: -0.02em;
    }

    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, rgba(247, 233, 221, 0.95) 0%, rgba(239, 231, 251, 0.92) 100%);
        border: 1px solid rgba(255,255,255,0.55);
        border-radius: 28px;
        padding: 2.6rem 2.3rem;
        box-shadow: var(--shadow);
        margin-bottom: 1.4rem;
    }

    .hero-badge {
        display: inline-block;
        font-size: 0.82rem;
        font-weight: 600;
        color: #7c5b46;
        background: rgba(255,255,255,0.65);
        padding: 0.42rem 0.8rem;
        border-radius: 999px;
        border: 1px solid rgba(124, 91, 70, 0.10);
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.55rem;
        color: #2b211a;
    }

    .hero-subtitle {
        font-size: 1.08rem;
        line-height: 1.75;
        color: #5f5248;
        max-width: 780px;
        margin-bottom: 1rem;
    }

    .hero-note {
        font-size: 0.92rem;
        color: #6f6258;
    }

    .glass-card {
        background: var(--card);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.55);
        border-radius: 24px;
        padding: 1.35rem;
        box-shadow: var(--shadow-soft);
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.55rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
        color: #1f2937;
    }

    .section-note {
        font-size: 0.95rem;
        color: var(--muted);
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    .step-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.78rem;
        font-weight: 700;
        color: #845b41;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.65rem;
    }

    .mini-card {
        background: rgba(255,255,255,0.72);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 1rem 1rem 0.9rem 1rem;
        box-shadow: var(--shadow-soft);
        height: 100%;
    }

    .metric-card {
        background: rgba(255,255,255,0.85);
        border: 1px solid rgba(31, 41, 55, 0.06);
        border-radius: 22px;
        padding: 1rem 1.1rem;
        box-shadow: var(--shadow-soft);
        min-height: 130px;
    }

    .metric-label {
        font-size: 0.88rem;
        color: var(--muted);
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
        color: #111827;
    }

    .metric-sub {
        font-size: 0.92rem;
        color: #6b7280;
        margin-top: 0.3rem;
        line-height: 1.5;
    }

    .result-banner {
        border-radius: 22px;
        padding: 1.15rem 1.2rem;
        margin-bottom: 1rem;
        border-left: 6px solid transparent;
    }

    .result-low {
        background: var(--low-bg);
        border-left-color: var(--low-border);
        color: #1f5b33;
    }

    .result-moderate {
        background: var(--moderate-bg);
        border-left-color: var(--moderate-border);
        color: #734f0f;
    }

    .result-high {
        background: var(--high-bg);
        border-left-color: var(--high-border);
        color: #7b1f1f;
    }

    .result-title {
        font-size: 1.15rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .result-copy {
        font-size: 1rem;
        line-height: 1.65;
    }

    .tag-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.8rem;
    }

    .tag {
        display: inline-block;
        font-size: 0.82rem;
        font-weight: 600;
        color: #6b4d3b;
        background: rgba(255,255,255,0.72);
        border: 1px solid rgba(107, 77, 59, 0.12);
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
    }

    .insight-box {
        background: #ffffff;
        border: 1px solid rgba(31,41,55,0.07);
        border-radius: 22px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.8rem;
        box-shadow: var(--shadow-soft);
    }

    .insight-title {
        font-size: 1.02rem;
        font-weight: 750;
        margin-bottom: 0.35rem;
        color: #1f2937;
    }

    .insight-copy {
        color: #4b5563;
        line-height: 1.65;
        font-size: 0.96rem;
    }

    .bullet-list {
        margin: 0.2rem 0 0 0;
        padding: 0;
        list-style: none;
    }

    .bullet-list li {
        position: relative;
        padding-left: 1.2rem;
        margin-bottom: 0.95rem;
        color: #374151;
        line-height: 1.65;
    }

    .bullet-list li::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0.65rem;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
    }

    .subtle-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(31,41,55,0.10), transparent);
        margin: 1rem 0 0.5rem 0;
    }

    .disclaimer {
        font-size: 0.88rem;
        color: #6b7280;
        line-height: 1.6;
        text-align: center;
        margin-top: 1.2rem;
    }

    .stButton > button {
        width: 100%;
        border-radius: 16px;
        border: none;
        background: linear-gradient(135deg, #d97757 0%, #b96de2 100%);
        color: white;
        font-weight: 700;
        padding: 0.85rem 1rem;
        box-shadow: 0 10px 20px rgba(185, 109, 226, 0.20);
    }

    .stButton > button:hover {
        filter: brightness(1.02);
    }

    .stTextInput > div > div,
    .stNumberInput > div > div,
    .stSelectbox > div > div,
    .stFileUploader > div,
    .stSlider,
    .stTabs [data-baseweb="tab-list"] {
        border-radius: 16px !important;
    }

    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
    }

    .stAlert {
        border-radius: 16px;
    }

    .preview-frame {
        border-radius: 24px;
        overflow: hidden;
        border: 1px solid rgba(31,41,55,0.08);
        box-shadow: var(--shadow-soft);
    }

    .pill-note {
        display: inline-block;
        font-size: 0.84rem;
        color: #6b7280;
        background: rgba(255,255,255,0.78);
        border: 1px solid rgba(31,41,55,0.06);
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        margin-top: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
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
            "The AQI CSV must contain either ['City', 'Country', 'PM2.5 AQI Value'] or ['city', 'country', 'pm25_aqi'] columns."
        )

    df["city"] = df["city"].astype(str).str.strip().str.lower()
    df["country"] = df["country"].astype(str).str.strip().str.lower()
    df = df.groupby(["city", "country"], as_index=False)["pm25_aqi"].mean()
    return df


aqi_df = load_aqi()
city_options = sorted(aqi_df["city"].dropna().astype(str).str.title().unique().tolist())

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


def describe_visible_score(score: float) -> str:
    if score < 0.33:
        return "Limited visible signs"
    elif score < 0.66:
        return "Noticeable visible signs"
    return "More pronounced visible signs"


def describe_exposure_score(score: float) -> str:
    if score < 0.33:
        return "Lower cumulative exposure"
    elif score < 0.66:
        return "Moderate cumulative exposure"
    return "Elevated cumulative exposure"


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
        return "Your uploaded image shows limited visible features associated with photoaging, and your lifestyle profile suggests a relatively protective pattern overall."
    elif risk_label == "Moderate":
        return "Your image shows some visible features associated with photoaging, and your lifestyle profile suggests moderate cumulative exposure over time."
    return "Your image shows stronger visible features associated with photoaging, and your lifestyle profile suggests elevated cumulative exposure that may deserve closer attention."


def get_result_summary(risk_label):
    if risk_label == "Low":
        return "Overall, your current skin presentation and environmental profile suggest a lower photoaging risk at this time."
    elif risk_label == "Moderate":
        return "Overall, your current skin presentation and lifestyle profile suggest a moderate level of photoaging risk."
    return "Overall, your current skin presentation and lifestyle profile suggest a higher level of photoaging risk."


def build_recommendations(hours, cigs, pollution_score, sunscreen):
    tips = []

    if sunscreen == "no":
        tips.append("Use broad spectrum sunscreen daily, especially before any extended outdoor exposure.")

    if hours >= 4:
        tips.append("Reduce long periods of direct sun exposure where possible, especially during midday hours.")

    if cigs > 0:
        tips.append("Smoking is associated with faster visible skin aging. Reducing or stopping may support healthier skin over time.")

    if pollution_score >= 0.6:
        tips.append("Higher pollution exposure may contribute to skin stress. Gentle cleansing and barrier-supportive skincare may be helpful.")

    if hours <= 2 and cigs == 0 and sunscreen == "yes" and pollution_score < 0.6:
        tips.append("Your current habits appear relatively protective. Staying consistent may help reduce long term photoaging risk.")

    return tips


def get_skin_routine(risk_label):
    if risk_label == "Low":
        return [
            "Use a gentle cleanser morning and evening.",
            "Apply sunscreen daily with broad spectrum SPF 30 or higher.",
            "Use a simple moisturizer to support skin barrier health.",
            "Maintain protective habits such as limiting unnecessary sun exposure.",
        ]
    elif risk_label == "Moderate":
        return [
            "Cleanse gently twice daily and avoid harsh scrubbing.",
            "Apply broad spectrum SPF 30 or higher every morning and reapply when outdoors for extended periods.",
            "Use a moisturizer consistently to support skin barrier health.",
            "Consider antioxidant based skincare to support protection against environmental stress.",
            "Reduce prolonged direct sun exposure where possible.",
        ]
    else:
        return [
            "Use a gentle, non irritating cleanser morning and evening.",
            "Apply broad spectrum SPF 50 every morning and reapply regularly when outdoors.",
            "Use a barrier supporting moisturizer daily.",
            "Consider antioxidant and pigment supportive skincare if appropriate.",
            "Minimize prolonged sun exposure and seek shade during peak UV hours.",
            "If you are concerned about visible skin changes, consult a dermatologist for individualized guidance.",
        ]


def predict_visible_photoaging(img_pil):
    img = img_pil.convert("RGB").resize(IMG_SIZE)
    img_np = np.array(img).astype(np.float32)
    img_batch = np.expand_dims(img_np, axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)
    pred = model.predict(x, verbose=0)
    return float(pred[0, 0]), img


def banner_class(risk_label):
    return {
        "Low": "result-low",
        "Moderate": "result-moderate",
        "High": "result-high",
    }[risk_label]


def build_html_list(items):
    return "<ul class='bullet-list'>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"


# ============================================================
# HERO
# ============================================================
st.markdown(
    """
    <div class='hero'>
        <div class='hero-badge'>AI supported skin exposure insight</div>
        <div class='hero-title'>Photoaging Check</div>
        <div class='hero-subtitle'>
            Upload a face image, complete a short lifestyle profile, and receive a refined estimate of visible photoaging risk with practical next steps.
        </div>
        <div class='hero-note'>Designed for educational and research use. This tool does not provide a medical diagnosis.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LAYOUT TOP
# ============================================================
left_col, right_col = st.columns([1.15, 0.85], gap="large")
selected_image = None

with left_col:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='step-chip'>Step 1</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Add your face image</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-note'>Use a clear frontal face image with good lighting and minimal background distractions for the most stable result.</div>",
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["Upload image", "Take a picture"])

    with tab1:
        uploaded_image = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            selected_image = uploaded_image

    with tab2:
        camera_image = st.camera_input("Capture a face image")
        if camera_image is not None:
            selected_image = camera_image

    if selected_image is not None:
        preview_image = Image.open(selected_image)
        st.markdown("<div class='pill-note'>Preview ready</div>", unsafe_allow_html=True)
        st.image(preview_image, caption="Selected image", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='step-chip'>Before you begin</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>How the estimate works</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='insight-box'>
            <div class='insight-title'>Image based signal</div>
            <div class='insight-copy'>A trained computer vision model estimates visible features associated with photoaging from your uploaded photo.</div>
        </div>
        <div class='insight-box'>
            <div class='insight-title'>Exposure profile</div>
            <div class='insight-copy'>Daily outdoor exposure, smoking behavior, sunscreen habits, and city level PM2.5 data are combined into a lifestyle and environment score.</div>
        </div>
        <div class='insight-box'>
            <div class='insight-title'>Final assessment</div>
            <div class='insight-copy'>The app blends visible image features with cumulative exposure indicators to produce a simple low, moderate, or high risk result.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# QUESTIONNAIRE
# ============================================================
with st.form("photoaging_form"):
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='step-chip'>Step 2</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Lifestyle questionnaire</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-note'>Answer a few brief questions so the model can combine visible skin features with your likely cumulative exposure profile.</div>",
        unsafe_allow_html=True,
    )

    q1, q2 = st.columns(2, gap="large")
    with q1:
        hours = st.slider(
            "How many hours do you usually spend outdoors during daylight each day?",
            0.0,
            8.0,
            2.0,
            0.5,
            help="Include commuting, outdoor work, exercise, and routine time spent outside during daylight.",
        )
        city = st.selectbox(
            "Which city do you live in most of the time?",
            options=city_options,
            index=city_options.index("London") if "London" in city_options else 0,
            help="This is used to estimate PM2.5 exposure from the available city dataset.",
        )

    with q2:
        cigs = st.number_input(
            "How many cigarettes do you smoke per day?",
            min_value=0,
            max_value=40,
            value=0,
            step=1,
        )
        sunscreen = st.selectbox(
            "Do you apply sunscreen daily?",
            ["yes", "no"],
        )

    st.markdown("<div class='subtle-divider'></div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Analyze My Photoaging Risk")
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# RUN
# ============================================================
if submitted:
    if selected_image is None:
        st.error("Please upload an image or take a picture before running the analysis.")
    elif not city.strip():
        st.error("Please select your city.")
    else:
        with st.spinner("Analyzing image and exposure profile..."):
            img_pil = Image.open(selected_image)
            visible_score, display_img = predict_visible_photoaging(img_pil)

            uv_score = clamp01(hours / 6.0)
            smoking_score = clamp01(cigs / 20.0)

            pm25_value, matched_country, city_matches = get_city_pm25(city)
            pollution_score = normalize_pm25(pm25_value)
            sunscreen_protection = 1.0 if sunscreen == "yes" else 0.0

            exposure_score = clamp01(
                0.70 * uv_score
                + 0.15 * smoking_score
                + 0.10 * pollution_score
                + 0.05 * (1 - sunscreen_protection)
            )

            final_score = clamp01(0.80 * visible_score + 0.20 * exposure_score)
            risk_label = category(final_score)
            risk_text = get_risk_text(risk_label)
            result_summary = get_result_summary(risk_label)
            tips = build_recommendations(hours, cigs, pollution_score, sunscreen)
            routine = get_skin_routine(risk_label)
            visible_label = describe_visible_score(visible_score)
            exposure_label = describe_exposure_score(exposure_score)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='step-chip'>Step 3</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Your result</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class='result-banner {banner_class(risk_label)}'>
            <div class='result-title'>{risk_label} Photoaging Risk</div>
            <div class='result-copy'>{risk_text}</div>
            <div class='tag-row'>
                <span class='tag'>Visible signal: {visible_label}</span>
                <span class='tag'>Exposure profile: {exposure_label}</span>
                <span class='tag'>Overall score: {final_score:.2f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3, gap="medium")
    with m1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>Visible skin change</div>
                <div class='metric-value'>{visible_score:.2f}</div>
                <div class='metric-sub'>{visible_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>Exposure profile</div>
                <div class='metric-value'>{exposure_score:.2f}</div>
                <div class='metric-sub'>{exposure_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>Overall assessment</div>
                <div class='metric-value'>{risk_label}</div>
                <div class='metric-sub'>{result_summary}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    content_left, content_right = st.columns([1.05, 0.95], gap="large")

    with content_left:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>What this means</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='section-note' style='margin-bottom:0;'>{result_summary} Photoaging refers to visible skin changes linked to long term environmental exposure, especially ultraviolet radiation. It may appear as wrinkles, dark spots, uneven tone, or reduced elasticity.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Personalized suggestions</div>", unsafe_allow_html=True)
        st.markdown(build_html_list(tips), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Suggested skin care routine</div>", unsafe_allow_html=True)
        st.markdown(build_html_list(routine), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with content_right:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Helpful skin health facts</div>", unsafe_allow_html=True)
        st.markdown(
            build_html_list([
                "Chronic sun exposure is the main external factor linked to photoaging.",
                "Visible photoaging can appear as wrinkles, dark spots, uneven tone, and texture changes.",
                "Consistent sunscreen use can help reduce cumulative UV related skin damage.",
                "Smoking and environmental pollution may contribute to long term skin stress.",
            ]),
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Pollution information</div>", unsafe_allow_html=True)

        if pm25_value is not None:
            if city_matches == 1:
                st.markdown(
                    f"<div class='section-note'><strong>Matched city data:</strong> {city.strip().title()}, {matched_country.title()}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='section-note'><strong>Matched city data from {city_matches} records:</strong> {city.strip().title()} across {matched_country}</div>",
                    unsafe_allow_html=True,
                )

            pollution_text = (
                "This pollution level suggests relatively lower environmental skin stress."
                if pollution_score < 0.33
                else "This pollution level suggests moderate environmental skin stress that may contribute over time."
                if pollution_score < 0.66
                else "This pollution level suggests elevated environmental skin stress that may meaningfully contribute over time."
            )
            st.markdown(
                f"<div class='section-note'><strong>Estimated PM2.5 AQI:</strong> {pm25_value:.1f}<br><br>{pollution_text}</div>",
                unsafe_allow_html=True,
            )
        else:
            suggestions = get_city_suggestions(city)
            st.warning("City pollution data was not found. A default pollution estimate was used.")
            if suggestions:
                st.markdown(build_html_list([f"Did you mean {s}?" for s in suggestions]), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Uploaded image</div>", unsafe_allow_html=True)
        st.image(display_img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='disclaimer'>This tool is intended for educational and research purposes only. It does not provide a medical diagnosis or replace professional dermatological advice.</div>",
        unsafe_allow_html=True,
    )

else:
    st.markdown(
        "<div class='disclaimer'>Complete the image upload and questionnaire above to generate your personalized photoaging insight.</div>",
        unsafe_allow_html=True,
    )
