
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

from gemini_explainer import explain_prediction
from gradcam import generate_gradcam_for_image


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Chest X-ray Classifier",
    page_icon="🩻",
    layout="wide",
)


# --------------------------------------------------
# 2. Load trained model
# --------------------------------------------------

MODEL_PATH = "models/mobilenet_best.keras"
IMAGE_SIZE = (160, 160)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# --------------------------------------------------
# 3. Application interface
# --------------------------------------------------

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
    type=["jpg", "jpeg", "png"],
)


# --------------------------------------------------
# 4. Process uploaded image
# --------------------------------------------------

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        # Clear the previous explanation when a different image is uploaded.
        current_image_id = (
            uploaded_file.name,
            uploaded_file.size,
        )

        if st.session_state.get("current_image_id") != current_image_id:
            st.session_state.pop("explanation", None)
            st.session_state["current_image_id"] = current_image_id

        left, right = st.columns(2)

        # Display uploaded image.
        with left:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)

        # ------------------------------------------
        # 5. Model prediction
        # ------------------------------------------

        resized = image.resize(IMAGE_SIZE)

        image_array = np.asarray(
            resized,
            dtype=np.float32,
        )

        image_array = np.expand_dims(image_array, axis=0)

        # Assumes the saved model includes any required preprocessing.
        prediction = float(
            model.predict(image_array, verbose=0)[0][0]
        )

        # Sigmoid output: PNEUMONIA class score.
        if prediction >= 0.5:
            predicted_class = "PNEUMONIA"
            confidence = prediction
        else:
            predicted_class = "NORMAL"
            confidence = 1.0 - prediction

        with right:
            st.subheader("Model Prediction")

            st.metric(
                "Predicted Dataset Class",
                predicted_class,
            )

            st.metric(
                "Model Confidence",
                f"{confidence:.2%}",
            )

            st.caption(
                "Confidence is an internal model score, not the probability "
                "that a person has or does not have a disease."
            )

        # ------------------------------------------
        # 6. Grad-CAM explainability
        # ------------------------------------------

        st.divider()
        st.subheader("Grad-CAM Explainability")

        try:
            with st.spinner("Generating Grad-CAM heatmap..."):
                grad_class, grad_confidence, overlay_image = (
                    generate_gradcam_for_image(image)
                )

            col_original, col_heatmap = st.columns(2)

            with col_original:
                st.image(
                    image,
                    caption="Original uploaded image",
                    use_container_width=True,
                )

            with col_heatmap:
                st.image(
                    overlay_image,
                    caption=(
                        f"Grad-CAM overlay — {grad_class} "
                        f"(model confidence: {grad_confidence:.2%})"
                    ),
                    use_container_width=True,
                )

            st.caption(
                "Highlighted regions indicate image features influencing "
                "the model's score. They do not establish disease or "
                "provide a medical diagnosis."
            )

        except Exception as error:
            st.error("Could not generate the Grad-CAM overlay.")
            st.caption(f"Error details: {error}")

        # ------------------------------------------
        # 7. Gemini educational explanation
        # ------------------------------------------

        st.divider()
        st.subheader("Gemini AI Educational Explanation")

        if st.button(
            "Generate Gemini Educational Explanation",
            key="generate_explanation",
        ):
            # Remove any old explanation before generating a new one.
            st.session_state.pop("explanation", None)

            with st.spinner("Generating educational explanation..."):
                try:
                    explanation = explain_prediction(
                        predicted_class,
                        confidence,
                    )

                    st.session_state["explanation"] = explanation

                except Exception as error:
                    st.error(
                        "Could not generate the explanation. "
                        "Check your Gemini API key, API access, "
                        "and internet connection."
                    )
                    st.caption(f"Error details: {error}")

        # Display only the explanation for the current uploaded image.
        if st.session_state.get("explanation"):
            st.markdown(st.session_state["explanation"])

            st.download_button(
                label="Download Educational Report",
                data=st.session_state["explanation"],
                file_name="chest_xray_educational_report.txt",
                mime="text/plain",
                key="download_explanation",
            )

    except Exception as error:
        st.error("Could not process the uploaded image.")
        st.caption(f"Error details: {error}")

else:
    st.info("Upload a chest X-ray image to begin.")
