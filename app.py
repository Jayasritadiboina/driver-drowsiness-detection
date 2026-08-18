import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

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

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 30px;
}

.card {
    padding: 25px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.result {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    margin-top: 20px;
}

.safe {
    background-color: #e8f5e9;
    color: #2e7d32;
}

.warning {
    background-color: #fff3cd;
    color: #856404;
}

.danger {
    background-color: #ffebee;
    color: #c62828;
}

.info-box {
    padding: 18px;
    border-radius: 12px;
    background-color: #eef4ff;
    margin-top: 15px;
}

.footer {
    text-align: center;
    color: #777;
    margin-top: 40px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "EfficientNet_B0_improved.keras"

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        return None

model = load_model()

# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Closed",
    "Open",
    "no_yawn",
    "yawn"
]

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🚗 AI Driver Drowsiness Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep Learning Based Real-Time Driver Safety Monitoring System'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System Information")

    st.write("**Model:** EfficientNet-B0")
    st.write("**Improved Model Accuracy:** 98.61%")
    st.write("**Classes:** 4")

    st.markdown("---")

    st.subheader("Detection Classes")

    st.write("👁️ Closed")
    st.write("👁️ Open")
    st.write("🥱 no_yawn")
    st.write("🥱 yawn")

    st.markdown("---")

    st.info(
        "Upload a driver's facial image to analyze "
        "eye and yawning conditions."
    )

# ============================================================
# MODEL CHECK
# ============================================================

if model is None:

    st.error(
        "❌ EfficientNet_B0_improved.keras is not available."
    )

    st.warning(
        "Place the trained model file "
        "`EfficientNet_B0_improved.keras` "
        "in the same folder as `app.py`."
    )

    st.stop()

# ============================================================
# MAIN COLUMNS
# ============================================================

left, right = st.columns(2)

# ============================================================
# IMAGE UPLOAD
# ============================================================

with left:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("📷 Driver Image")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Driver Image",
            use_container_width=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PREDICTION
# ============================================================

with right:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("🤖 AI Analysis")

    if uploaded_file is None:

        st.info(
            "Upload an image to start drowsiness detection."
        )

    else:

        if st.button(
            "🔍 Analyze Driver",
            use_container_width=True
        ):

            with st.spinner("AI model is analyzing the image..."):

                # ------------------------------------------------
                # PREPROCESS IMAGE
                # ------------------------------------------------

                img = image.resize((224, 224))

                img_array = np.array(img)

                img_array = img_array.astype("float32") / 255.0

                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                # ------------------------------------------------
                # PREDICTION
                # ------------------------------------------------

                predictions = model.predict(
                    img_array,
                    verbose=0
                )

                probabilities = predictions[0]

                predicted_index = np.argmax(probabilities)

                predicted_class = CLASS_NAMES[
                    predicted_index
                ]

                confidence = (
                    probabilities[predicted_index] * 100
                )

                # ------------------------------------------------
                # DISPLAY RESULT
                # ------------------------------------------------

                if predicted_class == "Closed":

                    st.markdown(
                        f"""
                        <div class="result danger">
                        😴 Eyes Closed<br>
                        <small>Confidence: {confidence:.2f}%</small>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.error(
                        "⚠️ Possible drowsiness detected. "
                        "Driver should stay alert or take a break."
                    )

                elif predicted_class == "yawn":

                    st.markdown(
                        f"""
                        <div class="result warning">
                        🥱 Yawning Detected<br>
                        <small>Confidence: {confidence:.2f}%</small>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.warning(
                        "⚠️ Yawning may indicate driver fatigue."
                    )

                elif predicted_class == "Open":

                    st.markdown(
                        f"""
                        <div class="result safe">
                        👁️ Eyes Open<br>
                        <small>Confidence: {confidence:.2f}%</small>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.success(
                        "✅ Driver appears alert."
                    )

                elif predicted_class == "no_yawn":

                    st.markdown(
                        f"""
                        <div class="result safe">
                        😊 No Yawning<br>
                        <small>Confidence: {confidence:.2f}%</small>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.success(
                        "✅ No yawning detected."
                    )

                # ------------------------------------------------
                # PROBABILITY TABLE
                # ------------------------------------------------

                st.markdown("---")

                st.subheader("📊 Prediction Confidence")

                for i, class_name in enumerate(CLASS_NAMES):

                    probability = (
                        probabilities[i] * 100
                    )

                    st.write(
                        f"**{class_name}** — "
                        f"{probability:.2f}%"
                    )

                    st.progress(
                        float(probabilities[i])
                    )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown("---")

st.subheader("🔄 How the System Works")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("### 📷")
    st.write("**Camera / Image**")
    st.caption("Driver image is captured")

with col2:
    st.markdown("### 🖼️")
    st.write("**Preprocessing**")
    st.caption("Image resized to 224×224")

with col3:
    st.markdown("### 🧠")
    st.write("**EfficientNet-B0**")
    st.caption("Deep learning analysis")

with col4:
    st.markdown("### 🔍")
    st.write("**Detection**")
    st.caption("Driver state identified")

with col5:
    st.markdown("### 🚨")
    st.write("**Safety Alert**")
    st.caption("Warning when required")

# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown("---")

st.subheader("📈 Model Performance")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "Test Accuracy",
        "98.61%"
    )

with m2:
    st.metric(
        "Precision",
        "98.62%"
    )

with m3:
    st.metric(
        "Recall",
        "98.60%"
    )

with m4:
    st.metric(
        "F1 Score",
        "98.60%"
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
    AI-Based Driver Drowsiness Detection System<br>
    Powered by EfficientNet-B0 Deep Learning Model
    </div>
    """,
    unsafe_allow_html=True
)
