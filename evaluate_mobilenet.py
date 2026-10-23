
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

DATA_DIR = Path("data/chest_xray")
MODEL_PATH = Path("models/mobilenet_best.keras")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "MobileNetV2 model not found. Run train_mobilenet.py first."
    )

model = tf.keras.models.load_model(MODEL_PATH)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="binary",
)

class_names = test_ds.class_names
print("Class mapping:", dict(enumerate(class_names)))

y_true = np.concatenate([
    labels.numpy().reshape(-1)
    for _, labels in test_ds
]).astype(int)

probabilities = model.predict(test_ds).reshape(-1)
y_pred = (probabilities >= 0.5).astype(int)

accuracy = accuracy_score(y_true, y_pred)

print(f"\nMobileNetV2 test accuracy: {accuracy:.4f}")
print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=class_names,
    zero_division=0,
))

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    output_dict=True,
    zero_division=0,
)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)
display.plot(cmap="Blues", values_format="d")
plt.title("MobileNetV2 - Test Confusion Matrix")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "mobilenet_confusion_matrix.png",
    dpi=150,
)
plt.show()

results = {
    "model": "MobileNetV2",
    "test_accuracy": float(accuracy),
    "classification_report": report,
    "confusion_matrix": cm.tolist(),
}

with open(OUTPUT_DIR / "mobilenet_results.json", "w") as file:
    json.dump(results, file, indent=2)

print("\nSaved: outputs/mobilenet_results.json")
print("Saved: outputs/mobilenet_confusion_matrix.png")
