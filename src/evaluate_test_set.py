from pathlib import Path
import argparse

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow import keras


parser = argparse.ArgumentParser()
parser.add_argument("videos_dir", type=Path)
parser.add_argument("baseline_model", type=Path)
parser.add_argument("temporal_model", type=Path)
args = parser.parse_args()

TARGET_WIDTH = 128
TARGET_HEIGHT = 64
TARGET_FPS = 1
SEQUENCE_LENGTH = 5
SMOOTHING_WINDOW = 5

test_videos = {
    "91": (602, 900),
    "14": (475, 600),
    "35": (106, 185),
}

baseline_model = keras.models.load_model(args.baseline_model)
temporal_model = keras.models.load_model(args.temporal_model)


def load_frames(video_path):
    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        raise RuntimeError(f"Nie udało się otworzyć: {video_path}")

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_step = max(1, round(fps / TARGET_FPS))

    frames = []
    times = []
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
            times.append(frame_number / fps)

        frame_number += 1

    video.release()

    return np.array(frames, dtype=np.float32), np.array(times)


def smooth(values):
    padding = SMOOTHING_WINDOW // 2
    padded = np.pad(values, (padding, padding), mode="edge")
    kernel = np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW
    result = np.convolve(padded, kernel, mode="valid")
    return result[:len(values)]


def normalize(values):
    mean = values.mean()
    std = values.std()

    if std == 0:
        return values - mean

    return (values - mean) / std


all_labels = []
all_baseline_scores = []
all_temporal_scores = []

for video_id, (anomaly_start, anomaly_end) in test_videos.items():
    video_path = args.videos_dir / f"{video_id}.mp4"

    frames, times = load_frames(video_path)

    baseline_reconstructed = baseline_model.predict(
        frames,
        batch_size=32,
        verbose=0
    )

    baseline_errors = np.mean(
        np.square(frames - baseline_reconstructed),
        axis=(1, 2, 3)
    )

    sequences = np.array(
        [
            frames[i:i + SEQUENCE_LENGTH]
            for i in range(len(frames) - SEQUENCE_LENGTH + 1)
        ],
        dtype=np.float32
    )

    temporal_reconstructed = temporal_model.predict(
        sequences,
        batch_size=16,
        verbose=0
    )

    temporal_errors = np.mean(
        np.square(sequences - temporal_reconstructed),
        axis=(1, 2, 3, 4)
    )

    comparison_times = times[SEQUENCE_LENGTH - 1:]
    baseline_errors = baseline_errors[SEQUENCE_LENGTH - 1:]

    baseline_scores = smooth(baseline_errors)
    temporal_scores = smooth(temporal_errors)

    labels = (
        (comparison_times >= anomaly_start)
        & (comparison_times <= anomaly_end)
    ).astype(int)

    baseline_auc = roc_auc_score(labels, baseline_scores)
    temporal_auc = roc_auc_score(labels, temporal_scores)

    print(f"{video_id}.mp4")
    print(f"Baseline AUC: {baseline_auc:.4f}")
    print(f"Temporal AUC: {temporal_auc:.4f}")
    print()

    all_labels.extend(labels)
    all_baseline_scores.extend(normalize(baseline_scores))
    all_temporal_scores.extend(normalize(temporal_scores))

all_labels = np.array(all_labels)
all_baseline_scores = np.array(all_baseline_scores)
all_temporal_scores = np.array(all_temporal_scores)

overall_baseline_auc = roc_auc_score(
    all_labels,
    all_baseline_scores
)

overall_temporal_auc = roc_auc_score(
    all_labels,
    all_temporal_scores
)

print("Łączny wynik dla zbioru ewaluacyjnego")
print(f"ROC AUC model jednoklatkowy: {overall_baseline_auc:.4f}")
print(f"ROC AUC model temporalny: {overall_temporal_auc:.4f}")