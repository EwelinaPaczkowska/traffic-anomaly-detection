from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras


parser = argparse.ArgumentParser()

parser.add_argument(
    "frames_path",
    type=Path
)

parser.add_argument(
    "model_path",
    type=Path
)

args = parser.parse_args()

frames = np.load(args.frames_path)
model = keras.models.load_model(args.model_path)

indices = [
    0,
    len(frames) // 4,
    len(frames) // 2,
    3 * len(frames) // 4,
    len(frames) - 1
]

samples = frames[indices]
reconstructed = model.predict(samples)

fig, axes = plt.subplots(2, len(samples), figsize=(15, 6))

for i in range(len(samples)):
    axes[0, i].imshow(samples[i].squeeze(), cmap="gray")
    axes[0, i].set_title("Oryginał")
    axes[0, i].axis("off")

    axes[1, i].imshow(reconstructed[i].squeeze(), cmap="gray")
    axes[1, i].set_title("Rekonstrukcja")
    axes[1, i].axis("off")

plt.tight_layout()
plt.show()