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

# IMPORTANT:
# These names/order must match your training dataset.
CLASS_NAMES = [
    "Closed",
    "Open",
    "no_yawn",
    "yawn"
]

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1250px;
        padding: 1.5rem 2rem 3rem;
    }

    .hero {
        padding: 35px;
        border-radius: 24px;
        background: linear-gradient(
            135deg,
            #0f172a,
            #1e3a5f,
            #0f766e
        );
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        margin: 0 0 10px 0;
        font-size: 2.5rem;
    }

    .hero p {
        color: #dbeafe;
        margin: 5px 0;
    }

    .card {
        padding: 24px;
        border: 1px solid #dbe4ef;
        border-radius: 20px;
        background: white;
        margin: 18px 0;
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <h1>🚗 DriveGuard AI</h1>

        <p>
        <b>
        AI Agent for Driver Drowsiness Detection and
        Intelligent Road Safety Assistance
        </b>
        </p>

        <p>
        EfficientNet-B0 • Drowsiness Detection • Safety Analytics
        </p>

    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading EfficientNet-B0...")
def load_drowsiness_model():

    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )


model = None

if not MODEL_PATH.exists():

    st.error(
        "❌ EfficientNet_B0.keras was not found."
    )

    st.info(
        "Upload the real trained EfficientNet_B0.keras "
        "file into the same GitHub folder as app.py."
    )

else:

    try:

        model = load_drowsiness_model()

    except Exception as e:

        st.error(
            "❌ EfficientNet_B0.keras could not be loaded."
        )

        st.code(str(e))

        st.stop()

# ============================================================
# PROBLEM STATEMENT
# ============================================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.header("🎯 Problem Statement")

st.write(
    "Develop an AI-assisted system for detecting visual "
    "signs of driver drowsiness and providing intelligent "
    "road-safety guidance."
)

st.write(
    "The current system uses EfficientNet-B0 to analyze "
    "driver images for eye-state and yawning-related "
    "drowsiness classes."
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "Model",
    "EfficientNet-B0"
)

c2.metric(
    "Input Size",
    "224 × 224"
)

c3.metric(
    "Output Classes",
    "4"
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# DROWSINESS DETECTION
# ============================================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.header(
    "🔍 Driver Drowsiness Detection"
)

st.write(
    "Upload a clear driver image or use your camera."
)

input_mode = st.radio(
    "Select input method",
    [
        "📁 Upload Image",
        "📷 Camera"
    ],
    horizontal=True
)

image = None

if input_mode == "📁 Upload Image":

    uploaded = st.file_uploader(
        "Upload JPG, JPEG or PNG",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded is not None:

        image = Image.open(
            uploaded
        ).convert("RGB")

else:

    camera = st.camera_input(
        "Take a driver image"
    )

    if camera is not None:

        image = Image.open(
            camera
        ).convert("RGB")


if image is not None:

    left, right = st.columns(2)

    with left:

        st.image(
            image,
            caption="Input Image",
            use_container_width=True
        )

    with right:

        st.subheader(
            "🧠 AI Detection Result"
        )

        if st.button(
            "🔍 Check Drowsiness",
            type="primary"
        ):

            img = image.resize(
                IMG_SIZE
            )

            img_array = np.asarray(
                img,
                dtype=np.float32
            )

            # IMPORTANT:
            # Keep this preprocessing consistent
            # with the preprocessing used during training.
            img_array = img_array / 255.0

            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            with st.spinner(
                "Analyzing image..."
            ):

                prediction = model.predict(
                    img_array,
                    verbose=0
                )[0]

            # ==================================================
            # BINARY MODEL
            # ==================================================

            if len(prediction) == 1:

                probability = float(
                    prediction[0]
                )

                if probability >= 0.5:

                    label = "Drowsy"
                    confidence = probability

                    st.error(
                        "😴 Possible Drowsiness Detected"
                    )

                else:

                    label = "Alert"
                    confidence = 1.0 - probability

                    st.success(
                        "😊 Driver appears Alert"
                    )

                st.metric(
                    "Prediction",
                    label
                )

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

            # ==================================================
            # MULTI-CLASS MODEL
            # ==================================================

            else:

                if len(prediction) != len(
                    CLASS_NAMES
                ):

                    st.error(
                        f"Model returned "
                        f"{len(prediction)} outputs, "
                        f"but this app expects "
                        f"{len(CLASS_NAMES)} classes."
                    )

                else:

                    class_index = int(
                        np.argmax(prediction)
                    )

                    label = CLASS_NAMES[
                        class_index
                    ]

                    confidence = float(
                        prediction[
                            class_index
                        ]
                    )

                    # ------------------------------------------
                    # RESULT
                    # ------------------------------------------

                    if label in [
                        "Closed",
                        "yawn"
                    ]:

                        st.error(
                            f"⚠️ Possible drowsiness: "
                            f"{label}"
                        )

                        st.warning(
                            "If the driver feels tired, "
                            "stop at a safe location and "
                            "take a break."
                        )

                    elif confidence < 0.60:

                        st.warning(
                            f"⚠️ Low-confidence prediction: "
                            f"{label}"
                        )

                    else:

                        st.success(
                            f"✅ Detected state: "
                            f"{label}"
                        )

                    r1, r2 = st.columns(2)

                    r1.metric(
                        "Prediction",
                        label
                    )

                    r2.metric(
                        "Confidence",
                        f"{confidence * 100:.2f}%"
                    )

                    # ------------------------------------------
                    # PROBABILITIES
                    # ------------------------------------------

                    st.subheader(
                        "📊 Prediction Probabilities"
                    )

                    results = sorted(
                        zip(
                            CLASS_NAMES,
                            prediction
                        ),
                        key=lambda x: float(x[1]),
                        reverse=True
                    )

                    for class_name, probability in results:

                        probability = float(
                            probability
                        )

                        st.write(
                            f"**{class_name}** — "
                            f"{probability * 100:.2f}%"
                        )

                        st.progress(
                            min(
                                max(
                                    probability,
                                    0.0
                                ),
                                1.0
                            )
                        )

else:

    st.info(
        "👆 Upload an image or use the camera "
        "to start detection."
    )

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# ROAD SAFETY
# ============================================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.header(
    "🚦 Intelligent Road Safety Assistance"
)

s1, s2, s3 = st.columns(3)

with s1:

    st.info(
        "**👁️ Driver Monitoring**\n\n"
        "Analyzes visual patterns represented "
        "in the trained drowsiness dataset."
    )

with s2:

    st.info(
        "**⚠️ Drowsiness Alert**\n\n"
        "Possible drowsiness states are highlighted "
        "with a safety warning."
    )

with s3:

    st.info(
        "**🛑 Safety Recommendation**\n\n"
        "If the driver feels tired, stop safely "
        "and take an appropriate break."
    )

st.warning(
    "The current EfficientNet-B0 model is a "
    "drowsiness model. Real traffic-signal recognition "
    "requires a separate traffic-signal dataset and model."
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# MODEL PERFORMANCE COMPARISON
# ============================================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.header(
    "📊 Model Performance Comparison"
)

st.write(
    "Previously reported project evaluation results:"
)

p1, p2, p3 = st.columns(3)

p1.metric(
    "CNN",
    "72.29%"
)

p2.metric(
    "MobileNetV3-Small",
    "84.53%"
)

p3.metric(
    "🏆 EfficientNet-B0",
    "90.53%"
)

st.table(
    {
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
)

st.subheader(
    "📈 Accuracy Comparison"
)

st.bar_chart(
    {
        "CNN": 72.29,
        "MobileNetV3-Small": 84.53,
        "EfficientNet-B0": 90.53
    }
)

st.caption(
    "90.53% is the reported model evaluation accuracy. "
    "An individual image's confidence is not the same "
    "as overall model accuracy."
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# PROJECT SUMMARY
# ============================================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.header(
    "📘 Project Summary"
)

x1, x2, x3, x4 = st.columns(4)

x1.metric(
    "Model",
    "EfficientNet-B0"
)

x2.metric(
    "Reported Accuracy",
    "90.53%"
)

x3.metric(
    "Classes",
    "4"
)

x4.metric(
    "System Status",
    "Online" if model is not None else "Offline"
)

st.write(
    "**Recognized Classes:** "
    "Closed • Open • no_yawn • yawn"
)

st.write(
    "**System Flow:** "
    "Image / Camera → Preprocessing → "
    "EfficientNet-B0 → Prediction → "
    "Confidence → Safety Guidance"
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <b>DriveGuard AI</b><br>

        AI Agent for Driver Drowsiness Detection and
        Intelligent Road Safety Assistance<br>

        Powered by EfficientNet-B0

    </div>
    """,
    unsafe_allow_html=True
)


