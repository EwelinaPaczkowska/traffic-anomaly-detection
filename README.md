# Traffic Anomaly Detection

Engineering thesis project focused on traffic anomaly detection in video using autoencoders.

## Dataset

The project uses the AI City Challenge 2021 Track 4 dataset.
Video files are not included in this repository.

## Setup

Create and activate a Python virtual environment, then install dependencies:

pip install -r requirements.txt

## Video inspection

To inspect a video and display selected frames:

python src/inspect_video.py "PATH_TO_VIDEO"

## Dataset summary

To generate a summary of the training dataset:

python src/dataset_summary.py "PATH_TO_TRAIN_DATA" "PATH_TO_TRAIN_ANOMALY_RESULTS"