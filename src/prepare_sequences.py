from pathlib import Path
import argparse

import cv2
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument(
    "videos_dir",
    type=Path
)

parser.add_argument(
    "--videos",
    nargs="+",
    default=["1", "3", "4"]
)

parser.add_argument(
    "--output",
    type=Path,
    default=Path("data/processed/train_sequences.npy")
)

args = parser.parse_args()

TARGET_WIDTH = 128
TARGET_HEIGHT = 64
TARGET_FPS = 1
SEQUENCE_LENGTH = 5

sequences = []

for video_id in args.videos:
    video_path = args.videos_dir / f"{video_id}.mp4"
    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        print(f"Nie udało się otworzyć: {video_path}")
        continue

    source_fps = video.get(cv2.CAP_PROP_FPS)
    frame_step = max(1, round(source_fps / TARGET_FPS))

    frame_number = 0
    buffer = []
    saved_sequences = 0

    while True:
        success, frame = video.read()

        if not success:
            break

        if frame_number % frame_step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
            frame = frame.astype(np.float32) / 255.0
            frame = np.expand_dims(frame, axis=-1)

            buffer.append(frame)

            if len(buffer) == SEQUENCE_LENGTH:
                sequences.append(np.stack(buffer))
                buffer = []
                saved_sequences += 1

        frame_number += 1

    video.release()

    print(f"{video_path.name}: zapisano {saved_sequences} sekwencji")

sequences = np.array(sequences, dtype=np.float32)

args.output.parent.mkdir(parents=True, exist_ok=True)
np.save(args.output, sequences)

print()
print(f"Gotowe: {args.output}")
print(f"Kształt danych: {sequences.shape}")
print(f"Typ danych: {sequences.dtype}")
print(f"Min: {sequences.min():.3f}")
print(f"Max: {sequences.max():.3f}")