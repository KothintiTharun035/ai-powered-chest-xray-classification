
# AI-Powered Chest X-ray Classification

An educational deep learning project that classifies chest X-ray images into the `NORMAL` and `PNEUMONIA` dataset classes using MobileNetV2, Grad-CAM explainability, and Google Gemini AI.

> **Disclaimer:** This project is for educational and research purposes only. It is not a clinically validated diagnostic system and must not be used to diagnose or rule out pneumonia.

## Features

- **Image classification:** Predicts NORMAL or PNEUMONIA using a trained MobileNetV2 model.
- **Baseline comparison:** Compares a custom CNN with transfer learning.
- **Model evaluation:** Reports accuracy, precision, recall, F1-score, and confusion matrices.
- **Explainable AI:** Uses Grad-CAM to visualize image regions influencing the model's score.
- **Generative AI:** Integrates Google Gemini to generate educational explanations and model limitations.
- **Interactive web application:** Upload images through a Streamlit interface.
- **Report download:** Download the generated educational explanation.

## Technology Stack

- Python
- TensorFlow / Keras
- MobileNetV2
- Scikit-learn
- NumPy and Pandas
- Matplotlib and Seaborn
- Streamlit
- Grad-CAM
- Google Gemini API

## Model Performance

Evaluation on the project's held-out test dataset produced the following results:

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1-score |
|---|---:|---:|---:|---:|
| Baseline CNN | 75.64% | 0.756 | 0.709 | 0.718 |
| MobileNetV2 | 88.62% | 0.877 | 0.896 | 0.882 |

MobileNetV2 improved test accuracy by 12.98 percentage points compared with the baseline CNN.

These results apply to the evaluated dataset. They do not establish clinical effectiveness or guarantee performance on images from other sources.

## Dataset

The project uses the Chest X-Ray Images (Pneumonia) dataset from Kaggle:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Download the dataset and place its extracted `chest_xray` folder inside `data/`.

Expected structure:

```text
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

The dataset is excluded from this Git repository. Follow the dataset's license and terms when using or redistributing it.

## Project Structure

```text
image-disease-classification/
├── app.py
├── gemini_explainer.py
├── gradcam.py
├── check_dataset.py
├── explore_dataset.py
├── train_baseline.py
├── evaluate_baseline.py
├── train_mobilenet.py
├── evaluate_mobilenet.py
├── compare_models.py
├── requirements.txt
├── .gitignore
├── README.md
├── models/
│   ├── baseline_best.keras
│   └── mobilenet_best.keras
└── outputs/
    ├── baseline_results.json
    ├── mobilenet_results.json
    ├── baseline_confusion_matrix.png
    ├── mobilenet_confusion_matrix.png
    ├── model_comparison.png
    └── gradcam_result.png
```

Generated files and additional output artifacts may vary.

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd image-disease-classification
```

Replace the placeholder with your actual repository URL.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use a Python version compatible with the TensorFlow version specified in the finalized dependencies.

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_gemini_api_key_here
```

Obtain an API key through Google AI Studio:

https://aistudio.google.com/apikey

Never commit `.env` or expose your API key in source code, screenshots, or public repositories.

### 5. Prepare the dataset

Download and extract the Kaggle dataset into `data/chest_xray/` using the directory structure above.

### 6. Run the application

```powershell
streamlit run app.py
```

Open the local address displayed in the terminal, typically:

http://localhost:8501

The saved MobileNetV2 model must be present at `models/mobilenet_best.keras`.

## How It Works

1. A user uploads a chest X-ray image.
2. MobileNetV2 produces a binary classification score.
3. The application displays the predicted dataset class and model confidence.
4. Grad-CAM generates a heatmap overlay to visualize influential image regions.
5. Gemini generates an educational explanation based on the model output.
6. The user can download the generated explanation.

## Limitations

- The model only distinguishes the two classes represented in its training data.
- Model confidence is not the probability that a patient has or does not have pneumonia.
- Predictions may be incorrect, including high-confidence errors.
- Dataset imbalance and differences in image acquisition can affect performance.
- Grad-CAM is an explanatory visualization, not proof of disease or a medical interpretation.
- Gemini's generated text is educational content and does not independently validate the prediction.
- Clinical decisions must be made by qualified healthcare professionals.

## Future Improvements

- Validate performance on independent datasets.
- Add model calibration and additional error analysis.
- Improve handling of class imbalance.
- Add automated tests and application monitoring.
- Deploy the Streamlit application with secure API-key configuration.

## Author

Tharun Kothinti

An educational AI/ML portfolio project demonstrating transfer learning, model evaluation, explainable AI, and generative AI integration.
