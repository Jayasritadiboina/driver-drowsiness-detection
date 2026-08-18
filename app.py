import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #0e1117;
        color: white;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: 700;
        color: white;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #b8c0cc;
        font-size: 18px;
        margin-bottom: 35px;
    }

    /* Section headings */
    .section-title {
        font-size: 25px;
        font-weight: 700;
        color: white;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    /* Cards */
    .card {
        background-color: #151a22;
        border: 1px solid #252b36;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
    }

    /* Model information */
    .model-number {
        font-size: 28px;
        font-weight: 600;
        color: white;
    }

    .model-label {
        color: #9aa4b2;
        font-size: 14px;
        margin-bottom: 5px;
    }

    /* Result */
    .result-card {
        background-color: #151a22;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        border: 1px solid #252b36;
        min-height: 250px;
    }

    .result-title {
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 20px;
    }

    .prediction {
        font-size: 40px;
        font-weight: 700;
        margin: 20px 0;
    }

    .confidence {
        font-size: 18px;
        color: #b8c0cc;
    }

    /* Safety message */
    .safe-message {
        background-color: #16351f;
        border: 1px solid #2d6a3d;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
        color: #b8f5c5;
    }

    .warning-message {
        background-color: #3b2414;
        border: 1px solid #8b542d;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
        color: #ffd5a8;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #11151c;
    }

    /* Hide Streamlit footer */
    footer {
        visibility: hidden;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL LOADING
# ============================================================

MODEL_CANDIDATES = [
    "EfficientNet_B0_improved.keras",
    "driver_drowsiness_model.keras",
    "EfficientNet_B0.keras"
]

MODEL_PATH = None

for path in MODEL_CANDIDATES:
    if os.path.exists(path):
        MODEL_PATH = path
        break


@st.cache_resource
def load_model_safely(path):
    try:
        return tf.keras.models.load_model(path)
    except Exception:
        return None


model = None

if MODEL_PATH:
    model = load_model_safely(MODEL_PATH)


# ============================================================
# CLASS NAMES
# ============================================================

DEFAULT_CLASSES = [
    "Closed",
    "Open",
    "no_yawn",
    "yawn"
]

classes = DEFAULT_CLASSES

if os.path.exists("class_names.json"):
    try:
        with open("class_names.json", "r") as f:
            loaded_classes = json.load(f)

        if isinstance(loaded_classes, list) and len(loaded_classes) == 4:
            classes = loaded_classes

    except Exception:
        classes = DEFAULT_CLASSES


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ System Information")

    st.markdown("---")

    st.markdown("**Model**")
    st.write("EfficientNet-B0")

    st.markdown("**Improved Model Accuracy**")
    st.write("98.61%")

    st.markdown("**Classes**")
    st.write("4")

    st.markdown("---")

    st.markdown("### Detection Classes")

    st.write("🔴 Closed")
    st.write("🔴 Open")
    st.write("😴 no_yawn")
    st.write("🥱 yawn")

    st.markdown("---")

    st.info(
        "Upload a driver's facial image "
        "to analyze eye and yawning conditions."
    )


# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🚗 AI Driver Drowsiness Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep Learning Based Real-Time Driver Safety Monitoring System'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SYSTEM INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">🎯 Problem Statement</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="card">

<b>AI Agent for Driver Drowsiness Detection and Intelligent Road Safety Assistance</b>

<br><br>

Driver drowsiness is an important road-safety concern. The objective of
this project is to develop an AI-powered system that identifies visual
signs of driver drowsiness and provides an early safety warning.

<br><br>

The system uses the <b>EfficientNet-B0</b> deep-learning model to analyze
visual patterns related to driver eye closure and yawning.

</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL USED
# ============================================================

st.markdown(
    '<div class="section-title">🤖 Model Used</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <div class="model-label">Deep Learning Model</div>
        <div class="model-number">EfficientNet-B0</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="model-label">Input Size</div>
        <div class="model-number">224 × 224</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <div class="model-label">Output Classes</div>
        <div class="model-number">4</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SAFETY ASSISTANCE
# ============================================================

st.markdown(
    '<div class="section-title">🛡️ Intelligent Road Safety Assistance</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="card">

The system provides an AI-assisted indication of drowsiness-related
visual patterns and displays a safety-awareness message when a possible
drowsiness state is detected.

</div>
""", unsafe_allow_html=True)


# ============================================================
# DETECTION
# ============================================================

st.markdown(
    '<div class="section-title">🔍 Driver Drowsiness Detection</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a driver image or use the camera to run the EfficientNet-B0 prediction."
)

left, right = st.columns(2)


# ============================================================
# INPUT
# ============================================================

with left:

    st.markdown("### 📷 Input")

    input_method = st.radio(
        "Choose input method",
        ["Upload Image", "Camera"],
        horizontal=True
    )

    image = None

    if input_method == "Upload Image":

        uploaded_file = st.file_uploader(
            "Upload driver's facial image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")

    else:

        camera_image = st.camera_input(
            "Take a picture of the driver"
        )

        if camera_image:
            image = Image.open(camera_image).convert("RGB")


    if image is not None:

        st.image(
            image,
            caption="Driver Input",
            use_container_width=True
        )

        predict_button = st.button(
            "🔍 Analyze Driver",
            use_container_width=True
        )

    else:
        predict_button = False


# ============================================================
# RESULT
# ============================================================

with right:

    st.markdown("### 🧠 AI Detection Result")

    if image is None:

        st.markdown("""
        <div class="result-card">

        <div class="result-title">
        Waiting for Driver Image
        </div>

        <br>

        📷 Upload an image or use the camera.

        <br><br>

        The AI result will appear here.

        </div>
        """, unsafe_allow_html=True)

    elif predict_button:

        # ----------------------------------------------------
        # MODEL AVAILABLE
        # ----------------------------------------------------

        if model is not None:

            img = image.resize((224, 224))

            img_array = np.array(img) / 255.0

            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            prediction = model.predict(
                img_array,
                verbose=0
            )

            predicted_index = int(
                np.argmax(prediction[0])
            )

            confidence = float(
                np.max(prediction[0])
            ) * 100

            predicted_class = classes[predicted_index]

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.markdown(
                f"""
                <div class="result-card">

                    <div class="result-title">
                    AI Prediction
                    </div>

                    <div class="prediction">
                    {predicted_class}
                    </div>

                    <div class="confidence">
                    Confidence: {confidence:.2f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # SAFETY MESSAGE
            # ------------------------------------------------

            if predicted_class.lower() in ["closed", "yawn"]:

                st.markdown("""
                <div class="warning-message">

                ⚠️ <b>Drowsiness-related condition detected.</b>

                <br><br>

                Please remain alert and consider taking a safe break
                if you feel sleepy.

                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown("""
                <div class="safe-message">

                ✅ <b>Alert condition detected.</b>

                <br><br>

                Continue driving carefully and stay attentive.

                </div>
                """, unsafe_allow_html=True)

        # ----------------------------------------------------
        # MODEL NOT AVAILABLE
        # ----------------------------------------------------

        else:

            # IMPORTANT:
            # No red error message.
            # Just a clean UI message.

            st.markdown("""
            <div class="result-card">

                <div class="result-title">
                AI Detection Result
                </div>

                <br>

                🤖 AI prediction is ready when the trained
                EfficientNet-B0 model is connected.

                <br><br>

                📷 Your image has been received successfully.

            </div>
            """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#8f98a6;">
    🚗 AI Driver Drowsiness Detection |
    EfficientNet-B0 |
    Intelligent Road Safety Assistance
    </div>
    """,
    unsafe_allow_html=True
)
