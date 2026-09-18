from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras


parser = argparse.ArgumentParser()

parser.add_argument(
    "sequences_path",
    type=Path
)

parser.add_argument(
    "model_path",
    type=Path
)

args = parser.parse_args()

sequences = np.load(args.sequences_path)
model = keras.models.load_model(args.model_path)

sample = sequences[0:1]

reconstructed = model.predict(sample, verbose=0)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i in range(5):
    axes[0, i].imshow(sample[0, i].squeeze(), cmap="gray")
    axes[0, i].set_title(f"Oryginał {i + 1}")
    axes[0, i].axis("off")

    axes[1, i].imshow(reconstructed[0, i].squeeze(), cmap="gray")
    axes[1, i].set_title(f"Rekonstrukcja {i + 1}")
    axes[1, i].axis("off")

plt.tight_layout()
plt.show()