from pathlib import Path
import argparse

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


parser = argparse.ArgumentParser()

parser.add_argument(
    "sequences_path",
    type=Path
)

parser.add_argument(
    "--model-output",
    type=Path,
    default=Path("results/temporal_autoencoder.keras")
)

args = parser.parse_args()

sequences = np.load(args.sequences_path)

inputs = keras.Input(shape=(5, 64, 128, 1))

x = layers.ConvLSTM2D(
    8,
    3,
    padding="same",
    return_sequences=True,
    activation="relu"
)(inputs)

x = layers.TimeDistributed(
    layers.MaxPooling2D(2, padding="same")
)(x)

x = layers.ConvLSTM2D(
    4,
    3,
    padding="same",
    return_sequences=True,
    activation="relu"
)(x)

x = layers.TimeDistributed(
    layers.MaxPooling2D(2, padding="same")
)(x)

x = layers.ConvLSTM2D(
    4,
    3,
    padding="same",
    return_sequences=True,
    activation="relu"
)(x)

x = layers.TimeDistributed(
    layers.UpSampling2D(2)
)(x)

x = layers.ConvLSTM2D(
    4,
    3,
    padding="same",
    return_sequences=True,
    activation="relu"
)(x)

x = layers.TimeDistributed(
    layers.UpSampling2D(2)
)(x)

x = layers.ConvLSTM2D(
    8,
    3,
    padding="same",
    return_sequences=True,
    activation="relu"
)(x)

outputs = layers.TimeDistributed(
    layers.Conv2D(
        1,
        3,
        padding="same",
        activation="sigmoid"
    )
)(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()

model.fit(
    sequences,
    sequences,
    epochs=10,
    batch_size=4,
    validation_split=0.2,
    shuffle=True
)

args.model_output.parent.mkdir(parents=True, exist_ok=True)
model.save(args.model_output)

print(f"Model zapisano w: {args.model_output}")