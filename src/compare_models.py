from pathlib import Path
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras


parser = argparse.ArgumentParser()

parser.add_argument("video_path", type=Path)
parser.add_argument("baseline_model", type=Path)
parser.add_argument("temporal_model", type=Path)
parser.add_argument("--anomaly-start", type=float, required=True)
parser.add_argument("--anomaly-end", type=float, required=True)
parser.add_argument(
    "--output",
    type=Path,
    default=Path("results/model_comparison.png")
)

args = parser.parse_args()

TARGET_WIDTH = 128
TARGET_HEIGHT = 64
TARGET_FPS = 1
SEQUENCE_LENGTH = 5
SMOOTHING_WINDOW = 5

baseline_model = keras.models.load_model(args.baseline_model)
temporal_model = keras.models.load_model(args.temporal_model)

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
frame_times = np.array(frame_times)

baseline_reconstructed = baseline_model.predict(
    frames,
    batch_size=32,
    verbose=0
)

baseline_errors = np.mean(
    np.square(frames - baseline_reconstructed),
    axis=(1, 2, 3)
)

sequences = []

for i in range(len(frames) - SEQUENCE_LENGTH + 1):
    sequences.append(
        frames[i:i + SEQUENCE_LENGTH]
    )

sequences = np.array(sequences, dtype=np.float32)

temporal_reconstructed = temporal_model.predict(
    sequences,
    batch_size=16,
    verbose=0
)

temporal_errors = np.mean(
    np.square(sequences - temporal_reconstructed),
    axis=(1, 2, 3, 4)
)

comparison_times = frame_times[SEQUENCE_LENGTH - 1:]
baseline_errors = baseline_errors[SEQUENCE_LENGTH - 1:]

def smooth(values):
    padding = SMOOTHING_WINDOW // 2

    padded = np.pad(
        values,
        (padding, padding),
        mode="edge"
    )

    kernel = np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW

    result = np.convolve(
        padded,
        kernel,
        mode="valid"
    )

    return result[:len(values)]


def normalize(values):
    mean = values.mean()
    std = values.std()

    if std == 0:
        return values - mean

    return (values - mean) / std


baseline_normalized = normalize(
    smooth(baseline_errors)
)

temporal_normalized = normalize(
    smooth(temporal_errors)
)

anomaly_mask = (
    (comparison_times >= args.anomaly_start)
    & (comparison_times <= args.anomaly_end)
)

normal_mask = ~anomaly_mask

baseline_anomaly_mean = baseline_errors[anomaly_mask].mean()
baseline_normal_mean = baseline_errors[normal_mask].mean()

temporal_anomaly_mean = temporal_errors[anomaly_mask].mean()
temporal_normal_mean = temporal_errors[normal_mask].mean()

baseline_ratio = baseline_anomaly_mean / baseline_normal_mean
temporal_ratio = temporal_anomaly_mean / temporal_normal_mean

print()
print("Model jednoklatkowy:")
print(f"Średni błąd podczas anomalii: {baseline_anomaly_mean:.6f}")
print(f"Średni błąd poza anomalią: {baseline_normal_mean:.6f}")
print(f"Stosunek anomalia/normalne: {baseline_ratio:.3f}")

print()
print("Model temporalny:")
print(f"Średni błąd podczas anomalii: {temporal_anomaly_mean:.6f}")
print(f"Średni błąd poza anomalią: {temporal_normal_mean:.6f}")
print(f"Stosunek anomalia/normalne: {temporal_ratio:.3f}")

args.output.parent.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(14, 6))

plt.plot(
    comparison_times,
    baseline_normalized,
    label="Model jednoklatkowy"
)

plt.plot(
    comparison_times,
    temporal_normalized,
    label="Model temporalny"
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
plt.title(f"Porównanie modeli — {args.video_path.name}")
plt.legend()
plt.tight_layout()
plt.savefig(args.output, dpi=150)
plt.show()

print(f"Liczba porównanych punktów: {len(comparison_times)}")
print(f"Średni błąd baseline: {baseline_errors.mean():.6f}")
print(f"Średni błąd temporalny: {temporal_errors.mean():.6f}")