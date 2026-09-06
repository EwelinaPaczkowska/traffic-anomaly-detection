from pathlib import Path
import argparse

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


parser = argparse.ArgumentParser()

parser.add_argument(
    "frames_path",
    type=Path
)

parser.add_argument(
    "--model-output",
    type=Path,
    default=Path("results/frame_autoencoder.keras")
)

args = parser.parse_args()

frames = np.load(args.frames_path)

inputs = keras.Input(shape=(64, 128, 1))

x = layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
x = layers.MaxPooling2D(2, padding="same")(x)

x = layers.Conv2D(8, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D(2, padding="same")(x)

x = layers.Conv2D(8, 3, activation="relu", padding="same")(x)

x = layers.UpSampling2D(2)(x)
x = layers.Conv2D(8, 3, activation="relu", padding="same")(x)

x = layers.UpSampling2D(2)(x)
x = layers.Conv2D(16, 3, activation="relu", padding="same")(x)

outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()

model.fit(
    frames,
    frames,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    shuffle=True
)

args.model_output.parent.mkdir(parents=True, exist_ok=True)
model.save(args.model_output)

print(f"Model zapisano w: {args.model_output}")