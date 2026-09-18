from pathlib import Path
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras


parser = argparse.ArgumentParser()
parser.add_argument("video_path", type=Path)
parser.add_argument("model_path", type=Path)
parser.add_argument(
    "--output",
    type=Path,
    default=Path("results/reconstruction_error.png")
)
parser.add_argument("--anomaly-start", type=float, required=True)
parser.add_argument("--anomaly-end", type=float, required=True)
args = parser.parse_args()

model = keras.models.load_model(args.model_path)

video = cv2.VideoCapture(str(args.video_path))

if not video.isOpened():
    raise RuntimeError(f"Nie udało się otworzyć: {args.video_path}")

fps = video.get(cv2.CAP_PROP_FPS)

errors = []
times = []

batch = []
batch_times = []

while True:
    success, frame = video.read()

    if not success:
        break

    frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES)) - 1

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (128, 64))
    frame = frame.astype(np.float32) / 255.0
    frame = np.expand_dims(frame, axis=-1)

    batch.append(frame)
    batch_times.append(frame_number / fps)

    if len(batch) == 128:
        batch_array = np.array(batch, dtype=np.float32)
        reconstructed = model.predict(batch_array, verbose=0)

        batch_errors = np.mean(
            np.square(batch_array - reconstructed),
            axis=(1, 2, 3)
        )

        errors.extend(batch_errors)
        times.extend(batch_times)

        batch = []
        batch_times = []

if batch:
    batch_array = np.array(batch, dtype=np.float32)
    reconstructed = model.predict(batch_array, verbose=0)

    batch_errors = np.mean(
        np.square(batch_array - reconstructed),
        axis=(1, 2, 3)
    )

    errors.extend(batch_errors)
    times.extend(batch_times)

video.release()

errors = np.array(errors)
times = np.array(times)

anomaly_start = args.anomaly_start
anomaly_end = args.anomaly_end

window_seconds = 5
window_size = max(1, int(fps * window_seconds))

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

normalized_errors = (smoothed_errors - mean_error) / std_error

args.output.parent.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(14, 6))

plt.plot(
    times,
    normalized_errors,
    label="Wygładzony i znormalizowany błąd"
)

plt.axvspan(
    anomaly_start,
    anomaly_end,
    alpha=0.2,
    label="Oznaczona anomalia"
)

plt.axvline(
    anomaly_start,
    linestyle="--"
)

plt.xlabel("Czas [s]")
plt.ylabel("Znormalizowany błąd rekonstrukcji")
plt.title(f"Błąd rekonstrukcji — {args.video_path.name}")
plt.legend()
plt.tight_layout()
plt.savefig(args.output, dpi=150)
plt.show()

print(f"Liczba przeanalizowanych klatek: {len(errors)}")
print(f"Średni błąd: {errors.mean():.6f}")
print(f"Maksymalny błąd: {errors.max():.6f}")
print(f"Wykres zapisano w: {args.output}")