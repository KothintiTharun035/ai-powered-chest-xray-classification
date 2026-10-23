# 🩻 AI-Powered Chest X-ray Classification

An AI-powered educational application that classifies chest X-ray images into **NORMAL** and **PNEUMONIA** dataset classes using MobileNetV2, Grad-CAM explainability, and Google Gemini AI.

The project combines deep learning, explainable AI (XAI), and generative AI in an interactive Streamlit web application.

> 

## ✨ Features

- **Chest X-ray Classification:** Classifies uploaded images into NORMAL or PNEUMONIA dataset categories.
- **Transfer Learning:** Uses pretrained MobileNetV2 for image classification.
- **Baseline Comparison:** Compares a custom CNN with MobileNetV2.
- **Model Evaluation:** Calculates accuracy, precision, recall, F1-score, and confusion matrices.
- **Grad-CAM Explainability:** Generates heatmap overlays to visualize image regions influencing the model's output.
- **Gemini AI Integration:** Generates educational explanations of model predictions and their limitations.
- **Interactive Web Interface:** Upload images and view predictions through Streamlit.
- **Downloadable Report:** Download the generated educational explanation as a text file.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Model Architecture | MobileNetV2, CNN |
| Image Processing | Pillow, NumPy |
| Data Analysis | Pandas, NumPy |
| Model Evaluation | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Explainable AI | Grad-CAM |
| Generative AI | Google Gemini API |
| Web Application | Streamlit |
| Environment Management | Python virtual environment, python-dotenv |
| Version Control | Git, GitHub |

## 📊 Model Performance

The models were evaluated on the project's held-out test dataset.

| Metric | Baseline CNN | MobileNetV2 |
|---|---:|---:|
| Test Accuracy | 75.64% | 88.62% |
| Macro Precision | 0.756 | 0.877 |
| Macro Recall | 0.709 | 0.896 |
| Macro F1-score | 0.718 | 0.882 |

**Key result:** MobileNetV2 improved test accuracy by **12.98 percentage points** compared with the baseline CNN.

These are results from the evaluated dataset, not evidence of clinical effectiveness or guaranteed performance on images from other sources.

## 🧠 How It Works

1. The user uploads a chest X-ray image through the Streamlit interface.
2. The image is resized to the input dimensions expected by the trained model.
3. MobileNetV2 produces a score for the PNEUMONIA class.
4. The application displays the predicted dataset class and model confidence.
5. Grad-CAM generates a heatmap overlay to visualize image features influencing the model's score.
6. Google Gemini generates an educational explanation based on the predicted class and confidence.
7. The user can download the generated explanation.

**Important:** Model confidence is an internal model score, not the probability that a patient has or does not have a disease. Grad-CAM is a visualization technique, not a medical interpretation.

## 🏗️ Project Structure

```text
ai-powered-chest-xray-classification/
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
├── README.md
├── .gitignore
├── models/
│   ├── baseline_best.keras
│   └── mobilenet_best.keras
├── outputs/
│   ├── baseline_results.json
│   ├── mobilenet_results.json
│   ├── baseline_history.json
│   ├── mobilenet_history.json
│   ├── baseline_confusion_matrix.png
│   ├── mobilenet_confusion_matrix.png
│   ├── model_comparison.png
│   └── gradcam_result.png
└── data/
    └── chest_xray/
        ├── train/
        ├── val/
        └── test/
```

The `data/` directory is excluded from the Git repository to avoid uploading the full dataset. Generated output files may vary.

## 📂 Dataset

This project uses the Chest X-Ray Images (Pneumonia) dataset available on Kaggle:

**Dataset link:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The dataset contains images organized into NORMAL and PNEUMONIA categories.

Download and extract the dataset so the directory structure is:

```text
data/chest_xray/
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

Follow the dataset's license and terms when using or redistributing its contents.

## 🚀 Installation and Setup

### Prerequisites

- Python compatible with the TensorFlow version in `requirements.txt`
- Git
- A Google Gemini API key for generating explanations

### 1. Clone the repository

```bash
git clone https://github.com/KothintiTharun035/ai-powered-chest-xray-classification.git
cd ai-powered-chest-xray-classification
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_gemini_api_key_here
```

Get an API key from Google AI Studio:

https://aistudio.google.com/apikey

Replace the placeholder with your own key.

**Security:** Never commit `.env` or expose your API key in source code, screenshots, or public repositories. The `.env` file should remain excluded by `.gitignore`.

### 5. Prepare the dataset

Download the Kaggle dataset and extract it into `data/chest_xray/` if you intend to run the training or dataset-analysis scripts.

The saved model files must be present in `models/` to run the application without retraining.

### 6. Launch the application

```powershell
streamlit run app.py
```

Open the local URL displayed in the terminal, typically:

http://localhost:8501

## ☁️ Deployment

The application can be deployed using Streamlit Community Cloud.

1. Push the project to GitHub.
2. Sign in at https://share.streamlit.io/ using GitHub.
3. Create an app from the repository.
4. Set the main file path to `app.py`.
5. Add the Gemini API key under the app's **Secrets** settings:

```toml
GEMINI_API_KEY = "your_actual_api_key"
```

6. Deploy and check the application logs if a build error occurs.

Never store the real API key in GitHub.

## ⚠️ Limitations

- The model distinguishes only the two dataset classes used during training.
- Predictions may be incorrect, including predictions made with high confidence.
- Model performance may be affected by class imbalance, image quality, dataset bias, and differences between training and real-world images.
- The model has not been established as clinically safe or validated for medical use.
- Grad-CAM visualizations do not prove the presence or absence of pneumonia.
- Gemini-generated explanations are educational text and do not independently validate predictions.
- The application must not be used to make medical decisions.

A qualified healthcare professional must interpret medical images and make clinical decisions.

## 🔮 Future Improvements

- Validate performance on independent datasets.
- Add automated tests and improved error handling.
- Evaluate calibration and classification errors.
- Improve handling of class imbalance.
- Add more detailed model evaluation visualizations.
- Improve deployment monitoring and application reliability.

## 👨‍💻 Author

**Tharun Kothinti**

GitHub: https://github.com/KothintiTharun035

---

⭐ If you find this project useful for learning about deep learning, explainable AI, and generative AI, consider starring the repository.
