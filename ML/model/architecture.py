"""CNN architecture for handwritten digit classification."""

from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10


def build_digit_cnn(
    learning_rate: float = 0.001,
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
) -> keras.Model:
    """Build and compile a compact CNN for MNIST digits."""
    inputs = keras.Input(shape=input_shape, name="digit_input")

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1")(inputs)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="dropout1")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.5, name="dropout2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="digit_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def model_summary_dict(model: keras.Model) -> dict[str, Any]:
    """Return a JSON-serializable architecture summary."""
    layers_info = []
    for layer in model.layers:
        layers_info.append(
            {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "output_shape": str(getattr(layer, "output_shape", None)),
            }
        )
    return {
        "name": model.name,
        "input_shape": list(INPUT_SHAPE),
        "num_classes": NUM_CLASSES,
        "total_params": int(model.count_params()),
        "layers": layers_info,
        "framework": f"tensorflow-{tf.__version__}",
    }
