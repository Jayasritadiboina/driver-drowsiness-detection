import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗"
)

st.title("🚗 Driver Drowsiness Detection")
st.write("Upload an image to check the driver's condition.")

# Model path
MODEL_PATH = Path(__file__).parent / "EfficientNet_B0.keras"

# Check whether model exists
if not MODEL_PATH.exists():
    st.error(
        f"❌ Model file not found: {MODEL_PATH.name}\n\n"
        "Make sure EfficientNet_B0.keras is in the same GitHub folder as app.py."
    )
    st.stop()

# Load model
@st.cache_resource
def load_drowsiness_model():
    return load_model(str(MODEL_PATH), compile=False)

try:
    model = load_drowsiness_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error("❌ EfficientNet_B0.keras could not be loaded.")
    st.code(str(e))
    st.stop()

# Upload image
uploaded_file = st.file_uploader(
    "📷 Upload driver's image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Prepare image
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    if st.button("🔍 Check Drowsiness"):

        with st.spinner("Analyzing..."):
            prediction = model.predict(img_array, verbose=0)

        st.write("Prediction:", prediction)

        # Handle binary classification
        if prediction.shape[-1] == 1:
            probability = float(prediction[0][0])

            if probability >= 0.5:
                st.error("😴 Drowsy")
            else:
                st.success("😊 Alert")

        # Handle 2-class classification
        elif prediction.shape[-1] == 2:
            class_index = int(np.argmax(prediction[0]))

            if class_index == 0:
                st.success("😊 Alert")
            else:
                st.error("😴 Drowsy")

        # Handle multiple classes
        else:
            class_index = int(np.argmax(prediction[0]))
            confidence = float(np.max(prediction[0])) * 100

            st.info(
                f"Predicted class: {class_index}\n\n"
                f"Confidence: {confidence:.2f}%"
            )
