
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from gemini_explainer import explain_prediction
from gradcam import generate_gradcam_for_image

st.set_page_config(
    page_title="AI Chest X-ray Classifier",
    page_icon="🩻",
    layout="wide"
)

MODEL_PATH = "models/mobilenet_best.keras"
IMAGE_SIZE = (160, 160)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("🩻 AI-Powered Chest X-ray Classification")
st.write(
    "An educational project using MobileNetV2, Grad-CAM, "
    "and Gemini AI to demonstrate image classification."
)

st.warning(
    "Research and educational use only. This model cannot diagnose "
    "pneumonia or rule it out. Consult a qualified healthcare professional."
)

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns(2)

    with left:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with right:
        st.subheader("Model Prediction")

        resized = image.resize(IMAGE_SIZE)
        image_array = np.array(resized, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        # MobileNetV2 preprocessing is included in the saved model.
        prediction = float(model.predict(image_array, verbose=0)[0][0])

        # The model's sigmoid output represents the PNEUMONIA class score.
        if prediction >= 0.5:
            predicted_class = "PNEUMONIA"
            confidence = prediction
        else:
            predicted_class = "NORMAL"
            confidence = 1.0 - prediction

        st.metric("Predicted Dataset Class", predicted_class)
        st.metric("Model Confidence", f"{confidence:.2%}")

        st.caption(
            "Confidence is an internal model score, not the probability "
            "that a person has or does not have a disease."
        )

        if st.button("Generate Gemini Educational Explanation"):
            st.session_state.pop("explanation", None)
            with st.spinner("Generating educational explanation..."):
                try:
                    explanation = explain_prediction(
                        predicted_class, confidence
                    )
                    st.session_state["explanation"] = explanation
                except Exception as error:
                    st.error(
                        "Could not generate the explanation. "
                        "Check your Gemini API key and internet connection."
                    )
                    st.caption(str(error))

    if st.session_state.get("explanation"):
        st.divider()
        st.subheader("Gemini AI Educational Explanation")
        st.markdown(st.session_state["explanation"])

        st.download_button(
            "Download Explanation",
            data=st.session_state["explanation"],
            file_name="chest_xray_educational_report.txt",
            mime="text/plain"
        )
    st.subheader("Grad-CAM Explainability")

try:
    grad_class, grad_confidence, overlay_image = (
        generate_gradcam_for_image(image)
    )

    st.image(
        overlay_image,
        caption=(
            f"Grad-CAM overlay — {grad_class} "
            f"(model confidence: {grad_confidence:.2%})"
        ),
        use_container_width=True,
    )

    st.caption(
        "Highlighted regions indicate image features influencing the "
        "model's score; they do not establish disease or show a diagnosis."
    )

except Exception as error:
    st.error("Could not generate the Grad-CAM overlay.")
    st.caption(str(error))