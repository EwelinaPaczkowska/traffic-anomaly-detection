import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Inspect sample frames from a video.")
parser.add_argument("video_path", type=Path, help="Path to the video file")
args = parser.parse_args()

VIDEO_PATH = args.video_path

video = cv2.VideoCapture(str(VIDEO_PATH))

if not video.isOpened():
    raise RuntimeError(f"Nie udało się otworzyć nagrania: {VIDEO_PATH}")

fps = video.get(cv2.CAP_PROP_FPS)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = frame_count / fps

print(f"Plik: {VIDEO_PATH.name}")
print(f"FPS: {fps}")
print(f"Liczba klatek: {frame_count}")
print(f"Rozdzielczość: {width}x{height}")
print(f"Długość: {duration:.2f} s")

video.release()

timestamps = [220, 235, 245, 255, 270, 300]

fig, axes = plt.subplots(2, 3, figsize=(15, 7))

for ax, timestamp in zip(axes.flatten(), timestamps):
    video = cv2.VideoCapture(str(VIDEO_PATH))

    video.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

    success, frame = video.read()

    if success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        minutes = timestamp // 60
        seconds = timestamp % 60

        ax.imshow(frame_rgb)
        ax.set_title(f"{minutes}:{seconds:02d}")
        ax.axis("off")

    video.release()

plt.tight_layout()
plt.show()