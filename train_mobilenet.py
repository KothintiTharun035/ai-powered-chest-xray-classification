
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models

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

# Create a proper validation split from training images.
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
print("Class mapping:", dict(enumerate(class_names)))

# Compute class weights using training labels only.
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
    i: float(weight)
    for i, weight in enumerate(weights)
}
print("Class weights:", class_weights)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Data augmentation is applied only during training.
augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
], name="data_augmentation")

# Load pretrained MobileNetV2.
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet",
)

# Freeze pretrained layers for initial transfer learning.
base_model.trainable = False

inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
x = augmentation(inputs)

# MobileNetV2 expects input values scaled to [-1, 1].
x = layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1,
)(x)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
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
        str(MODEL_DIR / "mobilenet_best.keras"),
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

# Save training history.
history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(OUTPUT_DIR / "mobilenet_history.json", "w") as file:
    json.dump(history_data, file, indent=2)

print("\nMobileNetV2 training completed.")
print("Model: models/mobilenet_best.keras")
print("History: outputs/mobilenet_history.json")
