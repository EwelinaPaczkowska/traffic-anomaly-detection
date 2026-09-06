from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(
    description="Display sample frames from prepared NumPy dataset."
)

parser.add_argument(
    "frames_path",
    type=Path,
    help="Path to prepared .npy file"
)

args = parser.parse_args()

frames = np.load(args.frames_path)

print(f"Kształt danych: {frames.shape}")
print(f"Typ danych: {frames.dtype}")
print(f"Min: {frames.min():.3f}")
print(f"Max: {frames.max():.3f}")

sample_indices = [
    0,
    len(frames) // 5,
    2 * len(frames) // 5,
    3 * len(frames) // 5,
    4 * len(frames) // 5,
    len(frames) - 1
]

fig, axes = plt.subplots(2, 3, figsize=(12, 6))

for ax, index in zip(axes.flatten(), sample_indices):
    ax.imshow(frames[index].squeeze(), cmap="gray")
    ax.set_title(f"Klatka {index}")
    ax.axis("off")

plt.tight_layout()
plt.show()