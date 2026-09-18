from pathlib import Path
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras


parser = argparse.ArgumentParser()

parser.add_argument("video_path", type=Path)
parser.add_argument("model_path", type=Path)
parser.add_argument("--anomaly-start", type=float, required=True)
parser.add_argument("--anomaly-end", type=float, required=True)
parser.add_argument(
    "--output",
    type=Path,
    default=Path("results/temporal_reconstruction_error.png")
)

args = parser.parse_args()

TARGET_WIDTH = 128
TARGET_HEIGHT = 64
TARGET_FPS = 1
SEQUENCE_LENGTH = 5

model = keras.models.load_model(args.model_path)

video = cv2.VideoCapture(str(args.video_path))

if not video.isOpened():
    raise RuntimeError(f"Nie udało się otworzyć: {args.video_path}")

source_fps = video.get(cv2.CAP_PROP_FPS)
frame_step = max(1, round(source_fps / TARGET_FPS))

frames = []
frame_times = []

frame_number = 0

while True:
    success, frame = video.read()

    if not success:
        break

    if frame_number % frame_step == 0:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        frame = frame.astype(np.float32) / 255.0
        frame = np.expand_dims(frame, axis=-1)

        frames.append(frame)
        frame_times.append(frame_number / source_fps)

    frame_number += 1

video.release()

frames = np.array(frames, dtype=np.float32)

sequences = []
times = []

for i in range(len(frames) - SEQUENCE_LENGTH + 1):
    sequences.append(frames[i:i + SEQUENCE_LENGTH])
    times.append(frame_times[i + SEQUENCE_LENGTH - 1])

sequences = np.array(sequences, dtype=np.float32)
times = np.array(times)

reconstructed = model.predict(
    sequences,
    batch_size=16,
    verbose=0
)

errors = np.mean(
    np.square(sequences - reconstructed),
    axis=(1, 2, 3, 4)
)

window_size = 5
padding = window_size // 2

padded_errors = np.pad(
    errors,
    (padding, padding),
    mode="edge"
)

kernel = np.ones(window_size) / window_size

smoothed_errors = np.convolve(
    padded_errors,
    kernel,
    mode="valid"
)

smoothed_errors = smoothed_errors[:len(errors)]

mean_error = smoothed_errors.mean()
std_error = smoothed_errors.std()

normalized_errors = (
    smoothed_errors - mean_error
) / std_error

args.output.parent.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(14, 6))

plt.plot(
    times,
    normalized_errors,
    label="Temporalny błąd rekonstrukcji"
)

plt.axvspan(
    args.anomaly_start,
    args.anomaly_end,
    alpha=0.2,
    label="Oznaczona anomalia"
)

plt.axvline(
    args.anomaly_start,
    linestyle="--"
)

plt.xlabel("Czas [s]")
plt.ylabel("Znormalizowany błąd rekonstrukcji")
plt.title(f"Model temporalny — {args.video_path.name}")
plt.legend()
plt.tight_layout()
plt.savefig(args.output, dpi=150)
plt.show()

print(f"Liczba sekwencji: {len(sequences)}")
print(f"Średni błąd: {errors.mean():.6f}")
print(f"Maksymalny błąd: {errors.max():.6f}")