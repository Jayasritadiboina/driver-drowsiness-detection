import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DriveGuard AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = Path(__file__).parent / "EfficientNet_B0.keras"
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "Closed",
    "Open",
    "no_yawn",
    "yawn"
]

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #0b0f14;
    color: white;
}

.block-container {
    max-width: 1200px;
    padding: 1rem 2rem 3rem;
}

/* Main title */

.main-title {
    text-align: center;
    font-size: 26px;
    font-weight: 800;
    color: white;
    margin-bottom: 18px;
}

/* White separator */

.separator {
    height: 16px;
    background: white;
    border-radius: 8px;
    margin: 18px 0 25px 0;
}

/* Section */

.section {
    background: #0b0f14;
    padding: 5px 0 10px 0;
}

/* Headers */

h1, h2, h3 {
    color: white !important;
}

.section-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 10px;
}

/* Cards */

.card {
    background: #111827;
    border: 1px solid #263244;
    border-radius: 8px;
    padding: 16px;
    min-height: 120px;
}

.card-title {
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 8px;
}

.card-text {
    color: #cbd5e1;
    font-size: 12px;
}

/* Metrics */

.metric {
    font-size: 22px;
    font-weight: 700;
    color: white;
}

.metric-label {
    color: #94a3b8;
    font-size: 11px;
}

/* Status */

.alert {
    background: #3f1d1d;
    border: 1px solid #ef4444;
    padding: 15px;
    border-radius: 8px;
    color: #fecaca;
}

.safe {
    background: #123524;
    border: 1px solid #22c55e;
    padding: 15px;
    border-radius: 8px;
    color: #bbf7d0;
}

.info-box {
    background: #102a43;
    border: 1px solid #1d4ed8;
    padding: 12px;
    border-radius: 7px;
    color: #dbeafe;
    font-size: 13px;
}

/* Footer */

.footer {
    text-align: center;
    color: #64748b;
    padding: 25px;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None

    try:
        return tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )
    except Exception:
        return None


model = load_model()

# ============================================================
# TITLE
# ============================================================

st.markdown(
    """
    <div class="main-title">
    🚗 AI Agent for Driver Drowsiness Detection and Intelligent
    <br>
    Road Safety Assistance using EfficientNet-B0
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

# ============================================================
# PROBLEM STATEMENT
# ============================================================

st.markdown(
    '<div class="section-title">🎯 Problem Statement</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="card">
    <b>AI Agent for Driver Drowsiness Detection and Intelligent Road Safety Assistance</b>
    <br><br>
    Driver drowsiness is an important road-safety concern.
    The objective of this project is to develop an AI-powered
    system that identifies visual signs of driver drowsiness
    and provides an early safety warning.
    <br><br>
    The system uses the EfficientNet-B0 deep-learning model
    to analyze visual patterns related to driver eye closure
    and yawning.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("### 🧠 Model Used")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="card">
        <div class="metric-label">Deep Learning Model</div>
        <div class="metric">EfficientNet-B0</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        """
        <div class="card">
        <div class="metric-label">Input Size</div>
        <div class="metric">224 × 224</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        """
        <div class="card">
        <div class="metric-label">Output Classes</div>
        <div class="metric">4</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("### 🛡️ Intelligent Road Safety Assistance")

st.write(
    "The system provides an AI-assisted indication of "
    "drowsiness-related visual patterns and displays a "
    "safety-awareness message when a possible drowsiness "
    "state is detected."
)

st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

# ============================================================
# MODEL STATUS
# ============================================================

if model is None:

    st.warning(
        "⚠️ EfficientNet_B0.keras is not available. "
        "The interface is running in UI mode. "
        "Add the trained model file to enable prediction."
    )

else:

    st.success("🟢 EfficientNet-B0 model loaded successfully.")

# ============================================================
# DROWSINESS DETECTION
# ============================================================

st.markdown(
    '<div class="section-title">🔍 Driver Drowsiness Detection</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a driver image or use the camera to run "
    "the EfficientNet-B0 prediction."
)

left, right = st.columns(2)

# ============================================================
# INPUT
# ============================================================

with left:

    st.markdown("### 📷 Input")

    input_type = st.radio(
        "Choose input method",
        [
            "📁 Upload Image",
            "📷 Camera"
        ],
        horizontal=True
    )

    image = None

    if input_type == "📁 Upload Image":

        uploaded_file = st.file_uploader(
            "Upload a JPG, JPEG or PNG image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

    else:

        camera_image = st.camera_input(
            "Take a driver image"
        )

        if camera_image:

            image = Image.open(
                camera_image
            ).convert("RGB"
            )

    if image:

        st.image(
            image,
            caption="Input Image",
            use_container_width=True
        )

    else:

        st.info(
            "👆 Upload an image or use the camera "
            "to start detection."
        )

# ============================================================
# PREDICTION
# ============================================================

with right:

    st.markdown("### 🧠 AI Detection Result")

    if image is None:

        st.info(
            "Prediction result will appear here."
        )

    elif model is None:

        st.warning(
            "⚠️ Model not found."
        )

        st.write(
            "Upload `EfficientNet_B0.keras` to enable "
            "real AI prediction."
        )

    else:

        # Resize
        resized_image = image.resize(IMG_SIZE)

        # Convert to numpy
        image_array = np.asarray(
            resized_image,
            dtype=np.float32
        )

        # Batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Prediction
        probabilities = model.predict(
            image_array,
            verbose=0
        )[0]

        prediction_index = int(
            np.argmax(probabilities)
        )

        prediction = CLASS_NAMES[
            prediction_index
        ]

        confidence = float(
            probabilities[
                prediction_index
            ]
        )

        # ====================================================
        # RESULT
        # ====================================================

        st.metric(
            "Prediction",
            prediction
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

        # Drowsiness logic

        if confidence < 0.60:

            st.warning(
                "⚠️ Low-confidence prediction. "
                "Please provide a clearer image."
            )

        elif prediction in ["Closed", "yawn"]:

            st.markdown(
                """
                <div class="alert">
                🚨 <b>DRIVER DROWSY</b>
                <br><br>
                Possible drowsiness detected.
                Please stop at a safe location and take a break.
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="safe">
                ✅ <b>DRIVER ALERT</b>
                <br><br>
                No strong visual indication of drowsiness detected.
                </div>
                """,
                unsafe_allow_html=True
            )

        # ====================================================
        # PROBABILITIES
        # ====================================================

        st.markdown("### 📊 Prediction Probabilities")

        results = sorted(
            zip(
                CLASS_NAMES,
                probabilities
            ),
            key=lambda x: x[1],
            reverse=True
        )

        for class_name, probability in results:

            st.write(
                f"**{class_name}** — "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                float(probability)
            )

st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">📊 Model Performance Comparison</div>',
    unsafe_allow_html=True
)

st.write(
    "Previously reported evaluation results from the project:"
)

p1, p2, p3 = st.columns(3)

with p1:
    st.metric(
        "CNN",
        "72.29%"
    )

with p2:
    st.metric(
        "MobileNetV3-Small",
        "84.53%"
    )

with p3:
    st.metric(
        "🏆 EfficientNet-B0",
        "90.53%"
    )

performance_data = {
    "Model": [
        "CNN",
        "MobileNetV3-Small",
        "EfficientNet-B0"
    ],
    "Accuracy": [
        "72.29%",
        "84.53%",
        "90.53%"
    ],
    "Precision": [
        "72.86%",
        "85.79%",
        "91.19%"
    ],
    "Recall": [
        "72.29%",
        "84.53%",
        "90.53%"
    ],
    "F1-Score": [
        "72.11%",
        "84.15%",
        "90.44%"
    ]
}

st.table(performance_data)

st.markdown("### 📈 Accuracy Comparison")

st.bar_chart(
    {
        "CNN": 72.29,
        "MobileNetV3-Small": 84.53,
        "EfficientNet-B0": 90.53
    }
)

st.success(
    "🏆 EfficientNet-B0 currently has the highest "
    "reported accuracy: 90.53%."
)

st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

# ============================================================
# ROAD SAFETY
# ============================================================

st.markdown(
    '<div class="section-title">🛡️ Intelligent Road Safety Assistance</div>',
    unsafe_allow_html=True
)

s1, s2, s3 = st.columns(3)

with s1:
    st.markdown(
        """
        <div class="card">
        👁️ <b>Visual Analysis</b>
        <br><br>
        <span class="card-text">
        The model analyzes visual patterns associated
        with the trained drowsiness classes.
        </span>
        </div>
        """,
        unsafe_allow_html=True
    )

with s2:
    st.markdown(
        """
        <div class="card">
        ⚠️ <b>Drowsiness Alert</b>
        <br><br>
        <span class="card-text">
        Possible drowsiness states are highlighted
        with a safety warning.
        </span>
        </div>
        """,
        unsafe_allow_html=True
    )

with s3:
    st.markdown(
        """
        <div class="card">
        🛑 <b>Safety Recommendation</b>
        <br><br>
        <span class="card-text">
        Drivers who feel tired should stop safely
        and take an appropriate break.
        </span>
        </div>
        """,
        unsafe_allow_html=True
    )

st.warning(
    "This is an AI-assisted academic project and should "
    "not be treated as a certified vehicle safety or "
    "emergency system."
)

st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

# ============================================================
# PROJECT SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">📘 Project Summary</div>',
    unsafe_allow_html=True
)

a, b, c, d = st.columns(4)

with a:
    st.metric(
        "Model",
        "EfficientNet-B0"
    )

with b:
    st.metric(
        "Reported Accuracy",
        "90.53%"
    )

with c:
    st.metric(
        "Classes",
        "4"
    )

with d:
    st.metric(
        "System Status",
        "Online" if model else "UI Mode"
    )

st.write(
    "**Recognized Classes:** "
    "Closed • Open • no_yawn • yawn"
)

st.write(
    "**System Flow:** "
    "Image / Camera → Image Preprocessing → "
    "EfficientNet-B0 → Prediction → "
    "Confidence → Safety Assistance"
)

st.markdown(
    """
    <div class="footer">
    🚗 DriveGuard AI | Driver Drowsiness Detection<br>
    AI-assisted academic project
    </div>
    """,
    unsafe_allow_html=True
)
