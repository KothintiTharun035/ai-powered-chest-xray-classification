
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models

# Reproducibility
SEED = 42
tf.keras.utils.set_random_seed(SEED)

DATA_DIR = Path("data/chest_xray")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 10

# Build training and validation datasets.
# The original validation folder has only 16 images,
# so create a better validation split from training data.
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)

class_names = train_ds.class_names
print("Class mapping:", class_names)

# Improve input pipeline performance.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Compute class weights from training labels only.
y_train = np.concatenate([
    labels.numpy().reshape(-1)
    for _, labels in train_ds
]).astype(int)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=y_train,
)

class_weights = {
    index: float(weight)
    for index, weight in enumerate(weights)
}
print("Class weights:", class_weights)

# Simple baseline CNN.
model = models.Sequential([
    layers.Input(shape=(*IMAGE_SIZE, 3)),
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(16, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_DIR / "baseline_best.keras"),
        monitor="val_loss",
        save_best_only=True,
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
)

# Save final training history.
history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(OUTPUT_DIR / "baseline_history.json", "w") as file:
    json.dump(history_data, file, indent=2)

print("\nTraining completed.")
print("Best model saved to models/baseline_best.keras")
print("Training history saved to outputs/baseline_history.json")
