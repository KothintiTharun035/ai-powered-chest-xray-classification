
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

MODEL_PATH = Path("models/mobilenet_best.keras")
DATA_DIR = Path("data/chest_xray/test")
OUTPUT_DIR = Path("outputs")

IMAGE_SIZE = (160, 160)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the model once when this module is imported.
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

base_model = next(
    (
        layer for layer in model.layers
        if isinstance(layer, tf.keras.Model)
        and any(
            isinstance(sub_layer, tf.keras.layers.Conv2D)
            for sub_layer in layer.layers
        )
    ),
    None,
)

if base_model is None:
    raise RuntimeError("Could not locate the nested MobileNetV2 model.")

conv_layers = [
    layer for layer in base_model.layers
    if isinstance(layer, tf.keras.layers.Conv2D)
]

last_conv_layer = conv_layers[-1]

base_grad_model = tf.keras.Model(
    inputs=base_model.inputs,
    outputs=[last_conv_layer.output, base_model.output],
)

print("Grad-CAM model loaded:", base_model.name)
print("Grad-CAM layer:", last_conv_layer.name)


def generate_gradcam_for_image(image_input):
    """
    Generate a Grad-CAM overlay for a PIL image or image file path.

    Returns:
        predicted_class: NORMAL or PNEUMONIA
        confidence: model confidence score, not disease probability
        overlay_image: PIL image containing the Grad-CAM overlay
    """

    # Accept either an uploaded PIL image or a file path.
    if isinstance(image_input, (str, Path)):
        original = Image.open(image_input).convert("RGB")
    else:
        original = image_input.convert("RGB")

    resized = original.resize(IMAGE_SIZE)

    image_array = np.asarray(resized, dtype=np.float32)
    input_tensor = tf.convert_to_tensor(image_array[np.newaxis, ...])

    base_index = model.layers.index(base_model)

    with tf.GradientTape() as tape:
        # Apply any preprocessing/augmentation layers before MobileNetV2.
        x = input_tensor

        for layer in model.layers[1:base_index]:
            x = layer(x, training=False)

        feature_maps, base_output = base_grad_model(x, training=False)

        # Apply the original classification head.
        x = base_output

        for layer in model.layers[base_index + 1:]:
            x = layer(x, training=False)

        pneumonia_score = x[0, 0]

        target_score = tf.where(
            pneumonia_score >= 0.5,
            pneumonia_score,
            1.0 - pneumonia_score,
        )

    gradients = tape.gradient(target_score, feature_maps)

    if gradients is None:
        raise RuntimeError("Gradients are unavailable for Grad-CAM.")

    channel_weights = tf.reduce_mean(gradients, axis=(1, 2))

    heatmap = tf.reduce_sum(
        feature_maps[0] * channel_weights[0],
        axis=-1,
    )

    heatmap = tf.maximum(heatmap, 0)
    heatmap = tf.math.divide_no_nan(
        heatmap, tf.reduce_max(heatmap)
    ).numpy()

    if not np.isfinite(heatmap).all():
        raise RuntimeError("Grad-CAM heatmap contains invalid values.")

    pneumonia_probability = float(pneumonia_score.numpy())
    predicted_index = int(pneumonia_probability >= 0.5)

    predicted_class = CLASS_NAMES[predicted_index]
    confidence = (
        pneumonia_probability
        if predicted_index == 1
        else 1.0 - pneumonia_probability
    )

    # Resize the heatmap to match the original image.
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        (original.height, original.width),
    ).numpy().squeeze()

    # Convert the heatmap into RGB colors.
    color_map = plt.get_cmap("jet")
    heatmap_rgb = (color_map(heatmap_resized)[..., :3] * 255).astype(
        np.uint8
    )
    heatmap_image = Image.fromarray(heatmap_rgb).convert("RGB")

    # Blend the heatmap over the original X-ray.
    overlay_image = Image.blend(
        original,
        heatmap_image,
        alpha=0.40,
    )

    return predicted_class, confidence, overlay_image


# Preserve the standalone test workflow.
def generate_gradcam(image_path):
    predicted_class, confidence, overlay_image = (
        generate_gradcam_for_image(image_path)
    )

    output_path = OUTPUT_DIR / "gradcam_result.png"
    overlay_image.save(output_path)

    print("\n--- Grad-CAM Results ---")
    print("Image:", image_path)
    print("Predicted class:", predicted_class)
    print(f"Model confidence: {confidence:.2%}")
    print("Heatmap saved:", output_path)
    print("Educational use only; not a medical diagnosis.")

    return predicted_class, confidence, output_path


if __name__ == "__main__":
    image_files = [
        path
        for class_name in CLASS_NAMES
        for path in sorted((DATA_DIR / class_name).glob("*"))
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    if not image_files:
        raise FileNotFoundError(f"No test images found under {DATA_DIR}")

    generate_gradcam(image_files[0])
