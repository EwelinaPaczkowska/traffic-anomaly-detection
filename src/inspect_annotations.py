from pathlib import Path
import argparse


parser = argparse.ArgumentParser(description="Inspect AI City anomaly annotations.")
parser.add_argument("annotations_path", type=Path, help="Path to train-anomaly-results.txt")
args = parser.parse_args()

annotations_path = args.annotations_path

with open(annotations_path, "r") as file:
    for line in file:
        video_id, start, end = map(int, line.split())

        start_min = start // 60
        start_sec = start % 60

        end_min = end // 60
        end_sec = end % 60

        print(
            f"{video_id}.mp4 -> "
            f"{start_min}:{start_sec:02d} - {end_min}:{end_sec:02d}"
        )