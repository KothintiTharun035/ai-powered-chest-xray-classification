
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("outputs")

with open(OUTPUT_DIR / "baseline_results.json") as file:
    baseline = json.load(file)

with open(OUTPUT_DIR / "mobilenet_results.json") as file:
    mobilenet = json.load(file)

models = [baseline, mobilenet]

print("\nMODEL COMPARISON")
print("-" * 75)
print(
    f"{'Model':<20} {'Accuracy':>10} "
    f"{'Macro Precision':>17} {'Macro Recall':>14} {'Macro F1':>10}"
)

for result in models:
    report = result["classification_report"]
    macro = report["macro avg"]

    print(
        f"{result['model']:<20} "
        f"{result['test_accuracy']:>9.2%} "
        f"{macro['precision']:>17.3f} "
        f"{macro['recall']:>14.3f} "
        f"{macro['f1-score']:>10.3f}"
    )

labels = [result["model"] for result in models]
accuracies = [
    result["test_accuracy"] * 100
    for result in models
]

plt.figure(figsize=(8, 5))
plt.bar(labels, accuracies)
plt.ylabel("Test Accuracy (%)")
plt.title("Baseline CNN vs MobileNetV2")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150)
plt.show()

print("\nComparison chart saved to outputs/model_comparison.png")
