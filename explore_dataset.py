
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATA_DIR = Path("data/chest_xray")
random.seed(42)

# Show example images from both classes.
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for row, class_name in enumerate(["NORMAL", "PNEUMONIA"]):
    folder = DATA_DIR / "train" / class_name

    images = [
        p for p in folder.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    selected = random.sample(images, min(4, len(images)))

    for col, image_path in enumerate(selected):
        with Image.open(image_path) as image:
            axes[row, col].imshow(image.convert("RGB"))

        axes[row, col].set_title(class_name)
        axes[row, col].axis("off")

plt.suptitle("Sample Chest X-ray Images")
plt.tight_layout()
plt.savefig("outputs_sample_images.png", dpi=150)
plt.show()

# Plot training class distribution.
classes = ["NORMAL", "PNEUMONIA"]
counts = [
    len([
        p for p in (DATA_DIR / "train" / cls).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    for cls in classes
]

plt.figure(figsize=(7, 4))
plt.bar(classes, counts)
plt.title("Training Dataset Class Distribution")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.savefig("outputs_class_distribution.png", dpi=150)
plt.show()

print("Image exploration completed.")
print("Training class counts:", dict(zip(classes, counts)))

