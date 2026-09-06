from pathlib import Path
import argparse

import cv2
import numpy as np


parser = argparse.ArgumentParser(
    description="Prepare video frames for autoencoder training."
)

parser.add_argument(
    "videos_dir",
    type=Path,
    help="Path to AI City training videos directory"
)

parser.add_argument(
    "--videos",
    nargs="+",
    default=["1", "3", "4"],
    help="Video IDs to process"
)

parser.add_argument(
    "--output",
    type=Path,
    default=Path("data/processed/train_frames.npy"),
    help="Output NumPy file"
)

args = parser.parse_args()


TARGET_WIDTH = 128
TARGET_HEIGHT = 64
TARGET_FPS = 1


frames = []

for video_id in args.videos:
    video_path = args.videos_dir / f"{video_id}.mp4"

    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        print(f"Nie udało się otworzyć: {video_path}")
        continue

    source_fps = video.get(cv2.CAP_PROP_FPS)

    frame_step = max(1, round(source_fps / TARGET_FPS))

    frame_number = 0
    saved_frames = 0

    while True:
        success, frame = video.read()

        if not success:
            break

        if frame_number % frame_step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frame = cv2.resize(
                frame,
                (TARGET_WIDTH, TARGET_HEIGHT)
            )

            frame = frame.astype(np.float32) / 255.0

            frame = np.expand_dims(frame, axis=-1)

            frames.append(frame)
            saved_frames += 1

        frame_number += 1

    video.release()

    print(
        f"{video_path.name}: zapisano {saved_frames} klatek"
    )


frames = np.array(frames, dtype=np.float32)

args.output.parent.mkdir(parents=True, exist_ok=True)

np.save(args.output, frames)

print()
print(f"Gotowe: {args.output}")
print(f"Kształt danych: {frames.shape}")
print(f"Typ danych: {frames.dtype}")
print(f"Min: {frames.min():.3f}")
print(f"Max: {frames.max():.3f}")