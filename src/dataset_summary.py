from pathlib import Path
import argparse
import csv

import cv2


parser = argparse.ArgumentParser(
    description="Create summary of AI City training videos."
)

parser.add_argument(
    "videos_dir",
    type=Path,
    help="Path to train-data directory"
)

parser.add_argument(
    "annotations_path",
    type=Path,
    help="Path to train-anomaly-results.txt"
)

parser.add_argument(
    "--output",
    type=Path,
    default=Path("results/train_dataset_summary.csv"),
    help="Output CSV path"
)

args = parser.parse_args()


# Wczytanie oficjalnych adnotacji anomalii
annotations = {}

with open(args.annotations_path, "r", encoding="utf-8") as file:
    for line in file:
        video_id, start, end = map(int, line.split())

        if video_id not in annotations:
            annotations[video_id] = []

        annotations[video_id].append((start, end))


# Sortowanie filmów numerycznie: 1, 2, 3... zamiast 1, 10, 100...
videos = sorted(
    args.videos_dir.glob("*.mp4"),
    key=lambda path: int(path.stem)
)

args.output.parent.mkdir(parents=True, exist_ok=True)


with open(args.output, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow([
        "video",
        "fps",
        "frame_count",
        "duration_seconds",
        "width",
        "height",
        "has_anomaly",
        "anomaly_count",
        "anomaly_intervals"
    ])

    for video_path in videos:
        video = cv2.VideoCapture(str(video_path))

        if not video.isOpened():
            print(f"Nie udało się otworzyć: {video_path.name}")
            continue

        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration = frame_count / fps if fps else 0

        video_id = int(video_path.stem)

        video_annotations = annotations.get(video_id, [])

        has_anomaly = "yes" if video_annotations else "no"

        intervals = "; ".join(
            f"{start}-{end}"
            for start, end in video_annotations
        )

        writer.writerow([
            video_path.name,
            round(fps, 2),
            frame_count,
            round(duration, 2),
            width,
            height,
            has_anomaly,
            len(video_annotations),
            intervals
        ])

        print(
            f"Przetworzono: {video_path.name} | "
            f"anomalia: {has_anomaly}"
        )

        video.release()


print(f"\nGotowe. Wyniki zapisano w: {args.output}")