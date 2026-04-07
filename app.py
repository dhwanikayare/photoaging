import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import difflib
import time


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Photoaging Insight AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# SESSION STATE
# ============================================================
DEFAULTS = {
    "app_screen": "landing",
    "selected_image_bytes": None,
    "selected_image_source": None,
    "analysis_complete": False,
    "result_payload": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================
# STYLING
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --bg: #f6f1eb;
        --card: rgba(255, 255, 255, 0.78);
        --card-strong: rgba(255,255,255,0.90);
        --text: #1f2937;
        --muted: #6b7280;
        --line: rgba(31, 41, 55, 0.08);
        --shadow-lg: 0 18px 40px rgba(64, 40, 18, 0.10);
        --shadow-md: 0 10px 26px rgba(64, 40, 18, 0.07);
        --shadow-sm: 0 6px 18px rgba(64, 40, 18, 0.05);
        --accent: #d97757;
        --accent-2: #8b5cf6;
        --success-bg: #eef9f1;
        --success-border: #4aa366;
        --warn-bg: #fff8e8;
        --warn-border: #d9a441;
        --danger-bg: #fdeeee;
        --danger-border: #d46666;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(217, 119, 87, 0.10), transparent 24%),
            radial-gradient(circle at top right, rgba(139, 92, 246, 0.10), transparent 28%),
            linear-gradient(180deg, #fcfaf7 0%, var(--bg) 100%);
        color: var(--text);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .fade-in {
        animation: fadeIn 0.8s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    h1, h2, h3, h4 {
        color: var(--text);
        letter-spacing: -0.02em;
    }

    .hero-shell {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, rgba(248, 234, 223, 0.92) 0%, rgba(238, 232, 251, 0.90) 100%);
        border: 1px solid rgba(255,255,255,0.58);
        border-radius: 32px;
        padding: 3.2rem 2.8rem;
        box-shadow: var(--shadow-lg);
        margin-bottom: 2rem;
    }

    .hero-badge {
        display: inline-block;
        font-size: 0.80rem;
        font-weight: 700;
        color: #7d5d48;
        background: rgba(255,255,255,0.72);
        border: 1px solid rgba(125,93,72,0.10);
        padding: 0.42rem 0.78rem;
        border-radius: 999px;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 3.15rem;
        font-weight: 850;
        line-height: 1.02;
        color: #2b211a;
        margin-bottom: 0.65rem;
    }

    .hero-copy {
        max-width: 760px;
        color: #5d534b;
        font-size: 1.08rem;
        line-height: 1.75;
        margin-bottom: 0.2rem;
    }

    .glass-card {
        background: rgba(255,255,255,0.88);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255,255,255,0.56);
        border-radius: 26px;
        padding: 1.35rem;
        box-shadow: var(--shadow-md);
        margin-bottom: 1rem;
    }

    .solid-card {
        background: rgba(255,255,255,0.96);
        border: 1px solid rgba(31,41,55,0.06);
        border-radius: 24px;
        padding: 1.2rem;
        box-shadow: var(--shadow-sm);
        margin-bottom: 0.6rem;
    }

    .section-kicker {
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: #8a6146;
        margin-bottom: 0.55rem;
    }

    .section-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 0.25rem;
    }

    .section-note {
        font-size: 0.96rem;
        color: var(--muted);
        line-height: 1.65;
        margin-bottom: 0.85rem;
    }

    .fact-card {
        background: rgba(255,255,255,0.80);
        border: 1px solid rgba(31,41,55,0.06);
        border-radius: 22px;
        padding: 1.1rem;
        box-shadow: var(--shadow-sm);
        height: 100%;
        min-height: 210px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }

    .fact-title {
        font-size: 1rem;
        font-weight: 750;
        color: #1f2937;
        margin-bottom: 0.35rem;
        min-height: 52px;
    }

    .fact-copy {
        font-size: 0.94rem;
        color: #596272;
        line-height: 1.65;
        flex-grow: 1;
    }

    .progress-wrap {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.6rem;
        margin-bottom: 1.2rem;
        flex-wrap: wrap;
    }

    .progress-step {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        background: rgba(255,255,255,0.60);
        border: 1px solid rgba(31,41,55,0.07);
        color: #7a7f87;
        padding: 0.52rem 0.85rem;
        border-radius: 999px;
        font-size: 0.90rem;
        font-weight: 700;
    }

    .progress-step.active {
        background: linear-gradient(135deg, rgba(217,119,87,0.14), rgba(139,92,246,0.14));
        color: #2c211a;
        border-color: rgba(217,119,87,0.18);
        box-shadow: var(--shadow-sm);
    }

    .progress-num {
        width: 24px;
        height: 24px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        background: rgba(255,255,255,0.85);
        font-size: 0.82rem;
        font-weight: 800;
    }

    .progress-arrow {
        color: #a3aab5;
        font-size: 1rem;
        font-weight: 700;
    }

    .option-card {
        background: rgba(255,255,255,0.84);
        border: 1px solid rgba(31,41,55,0.08);
        border-radius: 24px;
        padding: 1.1rem;
        box-shadow: var(--shadow-sm);
        min-height: 100%;
    }

    .option-title {
        font-size: 1.18rem;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 0.2rem;
    }

    .option-copy {
        font-size: 0.94rem;
        color: #6b7280;
        line-height: 1.6;
        margin-bottom: 0.8rem;
    }

    .ready-chip {
        display: inline-block;
        margin-top: 0.5rem;
        margin-bottom: 0.7rem;
        background: rgba(74,163,102,0.12);
        color: #22613a;
        border: 1px solid rgba(74,163,102,0.18);
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        font-size: 0.84rem;
        font-weight: 700;
    }

    .scanner-frame {
        position: relative;
        border-radius: 22px;
        overflow: hidden;
        border: 1px solid rgba(31,41,55,0.07);
        box-shadow: var(--shadow-sm);
        margin-top: 0.4rem;
        background: #0f172a;
    }

    .scanner-tint {
        position: absolute;
        inset: 0;
        pointer-events: none;
        background:
            linear-gradient(to bottom, rgba(15,23,42,0.00), rgba(15,23,42,0.10)),
            radial-gradient(circle at center, rgba(34,211,238,0.10), rgba(15,23,42,0.06));
        z-index: 1;
    }

    .scanner-overlay {
        position: absolute;
        inset: 0;
        pointer-events: none;
        background:
            linear-gradient(
                to bottom,
                rgba(255,255,255,0.00) 0%,
                rgba(34,211,238,0.00) 35%,
                rgba(34,211,238,0.28) 50%,
                rgba(34,211,238,0.00) 65%,
                rgba(255,255,255,0.00) 100%
            );
        animation: scanline 2s linear infinite;
        z-index: 3;
    }

    .scanner-face-box {
        position: absolute;
        top: 18%;
        left: 28%;
        width: 44%;
        height: 52%;
        border: 2px solid rgba(34,211,238,0.85);
        border-radius: 24px;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.08), 0 0 20px rgba(34,211,238,0.22);
        animation: scannerPulse 2s ease-in-out infinite;
        z-index: 4;
    }

    .scanner-corner {
        position: absolute;
        width: 34px;
        height: 34px;
        border-color: rgba(34,211,238,0.95);
        border-style: solid;
        animation: cornerPulse 1.5s ease-in-out infinite;
        z-index: 5;
        box-shadow: 0 0 12px rgba(34,211,238,0.24);
    }

    .scanner-corner.tl {
        top: calc(18% - 6px);
        left: calc(28% - 6px);
        border-width: 3px 0 0 3px;
        border-top-left-radius: 10px;
    }

    .scanner-corner.tr {
        top: calc(18% - 6px);
        left: calc(72% - 28px);
        border-width: 3px 3px 0 0;
        border-top-right-radius: 10px;
    }

    .scanner-corner.bl {
        top: calc(70% - 28px);
        left: calc(28% - 6px);
        border-width: 0 0 3px 3px;
        border-bottom-left-radius: 10px;
    }

    .scanner-corner.br {
        top: calc(70% - 28px);
        left: calc(72% - 28px);
        border-width: 0 3px 3px 0;
        border-bottom-right-radius: 10px;
    }

    .scanner-status {
        position: absolute;
        left: 50%;
        bottom: 18px;
        transform: translateX(-50%);
        padding: 0.42rem 0.85rem;
        border-radius: 999px;
        background: rgba(15,23,42,0.78);
        color: #d5f9ff;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        z-index: 6;
        border: 1px solid rgba(34,211,238,0.22);
        backdrop-filter: blur(6px);
    }

    @keyframes scanline {
        0% { transform: translateY(-100%); }
        100% { transform: translateY(100%); }
    }

    @keyframes scannerPulse {
        0%, 100% { opacity: 0.78; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.01); }
    }

    @keyframes cornerPulse {
        0%, 100% { opacity: 0.75; }
        50% { opacity: 1; }
    }

    .result-banner {
        border-radius: 24px;
        padding: 1.15rem 1.2rem;
        margin-bottom: 1rem;
        border-left: 6px solid transparent;
    }

    .result-low {
        background: var(--success-bg);
        border-left-color: var(--success-border);
        color: #1f5b33;
    }

    .result-moderate {
        background: var(--warn-bg);
        border-left-color: var(--warn-border);
        color: #734f0f;
    }

    .result-high {
        background: var(--danger-bg);
        border-left-color: var(--danger-border);
        color: #7b1f1f;
    }

    .result-title {
        font-size: 1.2rem;
        font-weight: 850;
        margin-bottom: 0.15rem;
    }

    .result-copy {
        font-size: 1rem;
        line-height: 1.7;
    }

    .metric-card {
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(31,41,55,0.06);
        border-radius: 22px;
        padding: 1rem 1.1rem;
        box-shadow: var(--shadow-sm);
        min-height: 170px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }

    .metric-label {
        font-size: 0.88rem;
        color: #6b7280;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 2rem;
        line-height: 1.05;
        font-weight: 850;
        color: #111827;
    }

    .metric-sub {
        font-size: 0.92rem;
        color: #6b7280;
        margin-top: 0.35rem;
        line-height: 1.5;
    }

    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.75rem;
    }

    .pill {
        display: inline-block;
        padding: 0.36rem 0.72rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
        background: rgba(255,255,255,0.76);
        color: #6b4d3b;
        border: 1px solid rgba(107,77,59,0.10);
    }

    .bullet-list {
        list-style: none;
        margin: 0.1rem 0 0 0;
        padding: 0;
    }

    .bullet-list li {
        position: relative;
        padding-left: 1.15rem;
        margin-bottom: 0.9rem;
        color: #374151;
        line-height: 1.7;
    }

    .bullet-list li::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0.68rem;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
    }

    .processing-stage {
        background: rgba(255,255,255,0.82);
        border: 1px solid rgba(31,41,55,0.06);
        border-radius: 20px;
        padding: 0.9rem 1rem;
        box-shadow: var(--shadow-sm);
        margin-bottom: 0.7rem;
        font-size: 0.98rem;
        color: #374151;
        line-height: 1.6;
    }

    .processing-stage.active {
        background: linear-gradient(135deg, rgba(217,119,87,0.11), rgba(139,92,246,0.11));
        border-color: rgba(217,119,87,0.15);
        font-weight: 700;
    }

    .disclaimer {
        text-align: center;
        color: #6b7280;
        font-size: 0.88rem;
        line-height: 1.65;
        margin-top: 1.2rem;
    }

    .stButton > button {
        width: 100%;
        border-radius: 18px;
        border: none;
        background: linear-gradient(135deg, #d97757 0%, #b96de2 100%);
        color: white;
        font-weight: 800;
        padding: 1rem 1.2rem;
        min-height: 56px;
        font-size: 1rem;
        box-shadow: 0 10px 20px rgba(185,109,226,0.20);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 26px rgba(0,0,0,0.12);
        filter: brightness(1.03);
    }

    .stTextInput > div > div,
    .stNumberInput > div > div,
    .stSelectbox > div > div,
    .stFileUploader > div,
    .stSlider,
    .stCameraInput > div {
        border-radius: 16px !important;
    }

    .stAlert {
        border-radius: 16px;
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
        return "Moderate visible signs"
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
        return "Your image currently shows limited visible features associated with photoaging, and your overall lifestyle profile appears relatively protective."
    elif risk_label == "Moderate":
        return "Your image shows some visible features associated with photoaging, and your current lifestyle profile suggests moderate cumulative exposure over time."
    return "Your image shows stronger visible features associated with photoaging, and your lifestyle profile suggests elevated cumulative exposure that may benefit from targeted changes."


def get_result_summary(risk_label):
    if risk_label == "Low":
        return "Overall, your current skin presentation and exposure profile suggest a lower level of photoaging risk at this time."
    elif risk_label == "Moderate":
        return "Overall, your current skin presentation and exposure profile suggest a moderate level of photoaging risk."
    return "Overall, your current skin presentation and exposure profile suggest a higher level of photoaging risk."


def build_recommendations(hours, cigs, pollution_score, sunscreen):
    tips = []

    if sunscreen == "no":
        tips.append("Use sunscreen daily before outdoor exposure, even on routine days with shorter time outside.")
    if hours >= 4:
        tips.append("Reduce extended direct sun exposure where possible, especially around midday when UV intensity is highest.")
    if cigs > 0:
        tips.append("Reducing or stopping smoking may support healthier skin over time and reduce visible aging stress.")
    if pollution_score >= 0.6:
        tips.append("Higher pollution exposure may contribute to skin stress, so gentle cleansing and barrier-supportive skincare may help.")
    if hours <= 2 and cigs == 0 and sunscreen == "yes" and pollution_score < 0.6:
        tips.append("Your current habits appear relatively protective, so consistency may help reduce long term photoaging risk.")

    return tips


def build_immediate_actions(hours, cigs, pollution_score, sunscreen):
    actions = []
    if sunscreen == "no":
        actions.append("Start using broad spectrum sunscreen daily.")
    if hours >= 4:
        actions.append("Cut down prolonged midday sun exposure when possible.")
    if cigs > 0:
        actions.append("Reduce daily smoking exposure.")
    if pollution_score >= 0.6:
        actions.append("Use cleansing and barrier-supportive skincare after high exposure days.")
    if not actions:
        actions.append("Maintain your current protective habits and stay consistent with sunscreen and sun avoidance practices.")
    return actions


def get_skin_routine(risk_label):
    if risk_label == "Low":
        return [
            "Use a gentle cleanser morning and evening.",
            "Apply broad spectrum SPF 30 or higher daily.",
            "Use a simple moisturizer to support barrier health.",
            "Continue limiting unnecessary direct sun exposure.",
        ]
    elif risk_label == "Moderate":
        return [
            "Cleanse gently twice daily and avoid harsh scrubbing.",
            "Apply broad spectrum SPF 30 or higher every morning and reapply when outdoors for longer periods.",
            "Use a moisturizer consistently to support barrier health.",
            "Consider antioxidant based skincare to support protection against environmental stress.",
            "Reduce prolonged direct sun exposure where possible.",
        ]
    else:
        return [
            "Use a gentle non-irritating cleanser morning and evening.",
            "Apply broad spectrum SPF 50 every morning and reapply regularly when outdoors.",
            "Use a barrier-supporting moisturizer daily.",
            "Consider antioxidant and pigment-supportive skincare if appropriate.",
            "Minimize prolonged sun exposure and seek shade during peak UV hours.",
            "If you are concerned about visible skin changes, consider consulting a dermatologist.",
        ]


def build_html_list(items):
    return "<ul class='bullet-list'>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"


def banner_class(risk_label):
    return {
        "Low": "result-low",
        "Moderate": "result-moderate",
        "High": "result-high",
    }[risk_label]


def progress_indicator(active_step: int):
    steps = [
        (1, "Image"),
        (2, "Lifestyle"),
        (3, "Results"),
    ]
    html_parts = ["<div class='progress-wrap'>"]
    for i, (num, label) in enumerate(steps):
        active_cls = "active" if num == active_step else ""
        html_parts.append(
            f"<div class='progress-step {active_cls}'><span class='progress-num'>{num}</span><span>{label}</span></div>"
        )
        if i < len(steps) - 1:
            html_parts.append("<div class='progress-arrow'>→</div>")
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def image_from_session():
    if st.session_state.selected_image_bytes is None:
        return None
    return Image.open(st.session_state.selected_image_bytes)


def save_uploaded_image(file_obj, source_label: str):
    if file_obj is not None:
        st.session_state.selected_image_bytes = file_obj
        st.session_state.selected_image_source = source_label


def get_scanner_mode():
    return st.session_state.selected_image_source if st.session_state.selected_image_source else "upload"


def render_scanner_preview(img, status_text="Scan complete"):
    source = get_scanner_mode()

    st.markdown("<div class='scanner-frame'>", unsafe_allow_html=True)
    st.image(img, use_container_width=True)
    st.markdown("<div class='scanner-tint'></div>", unsafe_allow_html=True)

    if source == "camera":
        st.markdown("<div class='scanner-overlay'></div>", unsafe_allow_html=True)
        st.markdown("<div class='scanner-face-box'></div>", unsafe_allow_html=True)
        st.markdown("<div class='scanner-corner tl'></div>", unsafe_allow_html=True)
        st.markdown("<div class='scanner-corner tr'></div>", unsafe_allow_html=True)
        st.markdown("<div class='scanner-corner bl'></div>", unsafe_allow_html=True)
        st.markdown("<div class='scanner-corner br'></div>", unsafe_allow_html=True)
        st.markdown("<div class='scanner-status'>Camera scan complete</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='scanner-overlay' style='opacity:0.75;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='scanner-face-box'></div>", unsafe_allow_html=True)
        st.markdown("<div class='scanner-corner tl'></div>", unsafe_allow_html=True)
        st.markdown("<div class='scanner-corner tr'></div>", unsafe_allow_html=True)
        st.markdown("<div class='scanner-corner bl'></div>", unsafe_allow_html=True)
        st.markdown("<div class='scanner-corner br'></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='scanner-status'>{status_text}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def predict_visible_photoaging(img_pil):
    img = img_pil.convert("RGB").resize(IMG_SIZE)
    img_np = np.array(img).astype(np.float32)
    img_batch = np.expand_dims(img_np, axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)
    pred = model.predict(x, verbose=0)
    return float(pred[0, 0]), img


def run_analysis(hours, cigs, city, sunscreen):
    img_pil = image_from_session()
    visible_score, display_img = predict_visible_photoaging(img_pil)

    uv_score = clamp01(hours / 6.0)
    smoking_score = clamp01(cigs / 20.0)

    pm25_value, matched_country, city_matches = get_city_pm25(city)
    pollution_score = normalize_pm25(pm25_value)
    sunscreen_protection = 1.0 if sunscreen == "yes" else 0.0

    exposure_score = clamp01(
        0.70 * uv_score + 0.15 * smoking_score + 0.10 * pollution_score + 0.05 * (1 - sunscreen_protection)
    )

    final_score = clamp01(0.80 * visible_score + 0.20 * exposure_score)
    risk_label = category(final_score)

    payload = {
        "visible_score": visible_score,
        "display_img": display_img,
        "exposure_score": exposure_score,
        "final_score": final_score,
        "risk_label": risk_label,
        "risk_text": get_risk_text(risk_label),
        "result_summary": get_result_summary(risk_label),
        "tips": build_recommendations(hours, cigs, pollution_score, sunscreen),
        "immediate_actions": build_immediate_actions(hours, cigs, pollution_score, sunscreen),
        "routine": get_skin_routine(risk_label),
        "pm25_value": pm25_value,
        "matched_country": matched_country,
        "city_matches": city_matches,
        "pollution_score": pollution_score,
        "visible_label": describe_visible_score(visible_score),
        "exposure_label": describe_exposure_score(exposure_score),
        "city": city,
    }
    return payload


def reset_analysis():
    st.session_state.app_screen = "landing"
    st.session_state.selected_image_bytes = None
    st.session_state.selected_image_source = None
    st.session_state.analysis_complete = False
    st.session_state.result_payload = None


def spacer(h=20):
    st.markdown(f"<div style='height:{h}px;'></div>", unsafe_allow_html=True)


# ============================================================
# LANDING SCREEN
# ============================================================
def render_landing():
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='hero-shell'>
            <div class='hero-badge'>AI-driven skin exposure analysis</div>
            <div class='hero-title'>Photoaging Insight AI</div>
            <div class='hero-copy'>
                Understand your skin exposure and photoaging risk using AI.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-kicker'>About this tool</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Built from dermatology and AI research</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-note'>This system combines deep learning based facial image analysis with environmental and lifestyle factors such as ultraviolet exposure, pollution, smoking, and sun protection habits to estimate photoaging risk. It is designed to make research-backed skin assessment more accessible without requiring specialized clinical equipment.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="large")
    facts = [
        (
            "Photoaging reflects cumulative UV damage",
            "Chronic ultraviolet exposure causes collagen breakdown, elastin damage, and pigmentation changes that become visible over time.",
        ),
        (
            "Visible skin changes are biologically meaningful",
            "Wrinkles, uneven tone, and texture variation often reflect underlying skin damage from long term environmental exposure.",
        ),
        (
            "Early detection supports prevention",
            "Visible photoaging can help raise early awareness of cumulative UV exposure and encourage protective skin habits.",
        ),
    ]

    for col, (title, copy) in zip([c1, c2, c3], facts):
        with col:
            st.markdown(
                f"""
                <div class='fact-card'>
                    <div class='fact-title'>{title}</div>
                    <div class='fact-copy'>{copy}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    spacer(14)
    cta_l, cta_c, cta_r = st.columns([1, 1.15, 1])
    with cta_c:
        analyze_cta = st.button("Analyze My Skin")
    if analyze_cta:
        with st.spinner("Preparing analysis..."):
            time.sleep(0.6)
        st.session_state.app_screen = "image_step"
        st.rerun()


# ============================================================
# IMAGE STEP
# ============================================================
def render_image_step():
    progress_indicator(active_step=1)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-kicker'>Step 1</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Add your face image</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-note'>Choose how you want to provide your image. For the most stable result, use a clear frontal face image with good lighting and minimal background distractions.</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(
            """
            <div class='option-card'>
                <div class='option-title'>Upload an image</div>
                <div class='option-copy'>Use an existing clear face photo from your device.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"], key="upload_step")
        if uploaded is not None:
            save_uploaded_image(uploaded, "upload")
            with st.spinner("Preparing scan..."):
                time.sleep(0.7)

    with right:
        st.markdown(
            """
            <div class='option-card'>
                <div class='option-title'>Use your camera</div>
                <div class='option-copy'>Capture a face image directly within the app for a more guided experience.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        camera = st.camera_input("Capture a face image", key="camera_step")
        if camera is not None:
            save_uploaded_image(camera, "camera")
            with st.spinner("Scanning face..."):
                time.sleep(0.8)

    img = image_from_session()
    if img is not None:
        st.markdown("<div class='ready-chip'>Face image ready for analysis</div>", unsafe_allow_html=True)
        render_scanner_preview(img, "Folder scan complete")

    spacer(16)
    nav_l, nav_c, nav_r = st.columns([1, 1.1, 1], gap="medium")
    with nav_l:
        back_clicked = st.button("Back")
    with nav_c:
        continue_disabled = img is None
        continue_clicked = st.button("Continue", disabled=continue_disabled)

    if back_clicked:
        st.session_state.app_screen = "landing"
        st.rerun()
    if continue_clicked:
        st.session_state.app_screen = "lifestyle_step"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# LIFESTYLE STEP
# ============================================================
def render_lifestyle_step():
    progress_indicator(active_step=2)

    st.markdown("<div class='glass-card fade-in' style='padding: 1.6rem 1.5rem;'>", unsafe_allow_html=True)
    st.markdown("<div class='section-kicker'>Step 2</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Lifestyle profile</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-note'>These questions help estimate your long term skin exposure patterns. Answer them based on your usual routine.</div>",
        unsafe_allow_html=True,
    )

    with st.form("lifestyle_form"):
        st.markdown("<div class='solid-card' style='padding: 1.2rem 1.2rem 0.4rem 1.2rem;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title' style='font-size:1.15rem; margin-bottom:0.15rem;'>Tell us about your typical exposure</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-note' style='margin-bottom:1rem;'>Your answers help combine visible facial features with daily environmental and lifestyle factors.</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="large")

        with c1:
            hours = st.slider(
                "How many hours do you usually spend outdoors during daylight each day?",
                min_value=0.0,
                max_value=8.0,
                value=0.0,
                step=0.5,
                help="Include commuting, exercise, outdoor work, and other regular daylight exposure.",
            )
            city = st.selectbox(
                "Which city do you live in most of the time?",
                options=["Select your city"] + city_options,
                index=0,
                help="This helps estimate environmental exposure using city level PM2.5 data.",
            )

        with c2:
            cigs = st.number_input(
                "How many cigarettes do you smoke per day?",
                min_value=0,
                max_value=40,
                value=0,
                step=1,
                help="Enter 0 if you do not smoke.",
            )
            sunscreen = st.selectbox(
                "Do you apply sunscreen daily?",
                ["Select an option", "yes", "no"],
                index=0,
                help="Daily sunscreen use is one of the most protective habits against photoaging.",
            )

        st.markdown("</div>", unsafe_allow_html=True)

        spacer(6)
        nav_l, nav_c, nav_r = st.columns([1, 1.1, 1], gap="medium")
        with nav_l:
            back_clicked = st.form_submit_button("Back")
        with nav_c:
            analyze_clicked = st.form_submit_button("Run Analysis")

    if back_clicked:
        st.session_state.app_screen = "image_step"
        st.rerun()

    if analyze_clicked:
        validation_errors = []
        if hours == 0.0:
            validation_errors.append("Please adjust your typical daylight exposure.")
        if city == "Select your city":
            validation_errors.append("Please select the city you live in most of the time.")
        if sunscreen == "Select an option":
            validation_errors.append("Please select whether you apply sunscreen daily.")

        if validation_errors:
            for msg in validation_errors:
                st.warning(msg)
        else:
            st.session_state.app_screen = "processing"
            st.session_state.analysis_inputs = {
                "hours": hours,
                "cigs": cigs,
                "city": city,
                "sunscreen": sunscreen,
            }
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# PROCESSING STEP
# ============================================================
def render_processing_step():
    progress_indicator(active_step=3)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-kicker'>Step 3</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Analyzing your profile</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-note'>Please wait while the system processes your facial image, lifestyle details, and environmental exposure profile.</div>",
        unsafe_allow_html=True,
    )

    placeholder = st.empty()
    stages = [
        "Processing facial image features...",
        "Evaluating lifestyle and environmental exposure...",
        "Generating personalized photoaging insights...",
    ]

    for idx, stage in enumerate(stages):
        html = ""
        for j, s in enumerate(stages):
            cls = "processing-stage active" if j == idx else "processing-stage"
            html += f"<div class='{cls}'>{s}</div>"
        placeholder.markdown(html, unsafe_allow_html=True)
        time.sleep(0.85)

    inputs = st.session_state.analysis_inputs
    payload = run_analysis(
        hours=inputs["hours"],
        cigs=inputs["cigs"],
        city=inputs["city"],
        sunscreen=inputs["sunscreen"],
    )
    st.session_state.result_payload = payload
    st.session_state.analysis_complete = True
    st.session_state.app_screen = "results"
    st.rerun()


# ============================================================
# RESULTS STEP
# ============================================================
def render_results():
    progress_indicator(active_step=3)
    result = st.session_state.result_payload

    st.markdown("<div class='glass-card fade-in' style='padding: 2.2rem 2rem 2rem 2rem; text-align:center;'>", unsafe_allow_html=True)
    st.markdown("<div class='section-kicker'>Your result</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title' style='margin-bottom:0.7rem;'>Photoaging assessment</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class='result-banner {banner_class(result['risk_label'])}' style='max-width: 920px; margin: 0 auto; text-align:left;'>
            <div class='result-title' style='font-size:1.5rem;'>{result['risk_label']} Photoaging Risk</div>
            <div class='result-copy'>{result['risk_text']}</div>
            <div class='pill-row'>
                <span class='pill'>Visible signs: {result['visible_label']}</span>
                <span class='pill'>Exposure level: {result['exposure_label']}</span>
                <span class='pill'>Score: {result['final_score']:.2f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    spacer(18)

    st.markdown("<div class='solid-card fade-in' style='padding: 1.3rem 1.4rem;'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title' style='font-size:1.3rem; margin-bottom:0.45rem;'>What you can do right now</div>", unsafe_allow_html=True)
    st.markdown(build_html_list(result["immediate_actions"][:2]), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    spacer(20)
    st.markdown("<div class='section-title' style='font-size:1.28rem; margin-bottom:0.85rem;'>Your skin profile</div>", unsafe_allow_html=True)
    overall_short = {
        "Low": "Lower overall risk profile",
        "Moderate": "Moderate overall risk profile",
        "High": "Higher overall risk profile",
    }[result["risk_label"]]

    m1, m2, m3 = st.columns(3, gap="medium")
    with m1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>Visible skin change</div>
                <div class='metric-value'>{result['visible_score']:.2f}</div>
                <div class='metric-sub'>{result['visible_label']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>Exposure profile</div>
                <div class='metric-value'>{result['exposure_score']:.2f}</div>
                <div class='metric-sub'>{result['exposure_label']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>Overall risk</div>
                <div class='metric-value'>{result['risk_label']}</div>
                <div class='metric-sub'>{overall_short}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    spacer(24)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("<div class='solid-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title' style='font-size:1.3rem;'>What this means</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='section-note' style='margin-bottom:0;'>Your results suggest relatively {result['exposure_label'].lower()} and {result['visible_label'].lower()}. Overall, this is consistent with your current risk profile.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        spacer(16)

        st.markdown("<div class='solid-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title' style='font-size:1.3rem;'>Suggested routine</div>", unsafe_allow_html=True)
        st.markdown(
            build_html_list([
                "Use a gentle cleanser morning and evening.",
                "Apply broad spectrum SPF 30 or higher daily.",
                "Use a moisturizer to support barrier health.",
            ]),
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='solid-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title' style='font-size:1.3rem;'>Key insights</div>", unsafe_allow_html=True)
        st.markdown(
            build_html_list([
                "Long term sun exposure is the main driver of visible photoaging.",
                "Early protection can reduce future skin damage.",
                "Consistent protective habits matter most.",
            ]),
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        spacer(16)

        st.markdown("<div class='solid-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title' style='font-size:1.3rem;'>Environmental exposure</div>", unsafe_allow_html=True)
        if result["pm25_value"] is not None:
            pollution_text = (
                "Based on your location, environmental exposure levels appear relatively low and are unlikely to significantly increase skin stress at this time."
                if result["pollution_score"] < 0.33
                else "Based on your location, environmental exposure may contribute moderately to long term skin stress over time."
                if result["pollution_score"] < 0.66
                else "Based on your location, environmental exposure appears elevated and may meaningfully contribute to long term skin stress."
            )
            st.markdown(
                f"<div class='section-note' style='margin-bottom:0;'><strong>{result['city'].title()}</strong><br><br>Estimated PM2.5 AQI: <strong>{result['pm25_value']:.1f}</strong><br><br>{pollution_text}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='section-note' style='margin-bottom:0;'>Environmental exposure data was not available for the selected city, so a default estimate was used.</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    spacer(26)
    btn_l, btn_c, btn_r = st.columns([1, 1.15, 1])
    with btn_c:
        if st.button("Start New Analysis"):
            reset_analysis()
            st.rerun()

    st.markdown(
        "<div class='disclaimer'>This tool is intended for educational and research purposes only. It does not provide a medical diagnosis or replace professional dermatological advice.</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# ROUTER
# ============================================================
screen = st.session_state.app_screen

if screen == "landing":
    render_landing()
elif screen == "image_step":
    render_image_step()
elif screen == "lifestyle_step":
    render_lifestyle_step()
elif screen == "processing":
    render_processing_step()
elif screen == "results":
    render_results()
else:
    render_landing()
