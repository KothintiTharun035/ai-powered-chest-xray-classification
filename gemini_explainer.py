
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing from your .env file."
    )

client = genai.Client(api_key=api_key)


def explain_prediction(predicted_class, confidence):
    """
    Generate an educational explanation of the
    classifier's output using Google Gemini.
    """

    prompt = f"""
    You are an educational AI assistant for a research
    project called AI-Powered Chest X-ray Classification.

    The image classifier returned:
    Predicted dataset class: {predicted_class}
    Model confidence for the predicted class: {confidence:.2%}

    Write a clear report with these sections:

    1. Prediction Summary
       Explain what the model output means.

    2. Educational Information
       Give general educational information about the
       predicted category.

    3. Model Confidence
       Explain why model confidence is not the probability
       that a patient has or does not have a disease.

    4. Limitations
       Explain possible errors, dataset bias, image quality
       issues, and differences between training and real-world
       medical images.

    5. Recommended Next Step
       State that a qualified healthcare professional must
       interpret medical images and make clinical decisions.

    Important safety rules:
    - This is not a medical diagnosis.
    - Do not invent symptoms, image findings, or patient history.
    - Do not claim to have inspected the X-ray.
    - Do not recommend treatment or medication.
    - Clearly state that the model may be wrong.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty explanation."
        )

    return response.text


if __name__ == "__main__":
    # Temporary connection test.
    # The Streamlit application will pass real predictions.
    explanation = explain_prediction("NORMAL", 0.8417)
    print("\n--- Gemini Educational Explanation ---\n")
    print(explanation)
