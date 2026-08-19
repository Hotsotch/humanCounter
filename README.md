# humanCounter

Real-time people counting from video using computer vision. The repo contains two counting approaches built at different stages of the project: a classic OpenCV background-subtraction tracker, and a newer YOLOv8-based detector with region-of-interest (ROI) counting.

Originally based on [epcm18/PeopleCounting-ComputerVision](https://github.com/epcm18/PeopleCounting-ComputerVision), extended and reworked from there.

## Approaches

### 1. YOLOv8 ROI counter — `countingYolov8.py` (current)
Runs a YOLOv8 model (`yolov8s.pt`) on each frame, filters detections to the `person` class, and counts how many detected people have their bounding-box center inside a defined ROI rectangle. Draws bounding boxes (green if inside the ROI, red if outside), the ROI outline, and a live count overlay.

### 2. Background subtraction line-crossing counter — `counter.py` + `Person.py` (legacy)
An earlier approach using `cv2.createBackgroundSubtractorMOG2` to detect motion blobs, then a simple centroid tracker (`Person.py`) to follow each blob across frames and count line crossings as "up" or "down." No ML model required — lighter weight, but less robust to lighting changes and multiple overlapping people.

## Repo structure

```
humanCounter/
├── countingYolov8.py   # YOLOv8 + ROI counter (current approach)
├── counter.py           # Background-subtraction + line-crossing counter (legacy)
├── Person.py             # Centroid tracker class used by counter.py
├── coco.names            # COCO class labels (reference list)
├── yolov8s.pt             # YOLOv8 small pretrained weights
└── requirements.txt
```

## Requirements

- Python 3.10+
- A webcam or video file to test against
- Please add your own video to track

Install dependencies:

```bash
pip install -r requirements.txt
```

Note: `requirements.txt` was generated from a CUDA-enabled environment (`torch==2.9.1+cu126`, `torchvision==0.24.1+cu126`). If you're on CPU-only or a different CUDA version, install PyTorch separately first following the instructions at [pytorch.org](https://pytorch.org/get-started/locally/), then install the rest of `requirements.txt`.

## Usage

### YOLOv8 counter

```bash
python countingYolov8.py
```

By default it reads from a video file (`test2_walking.mp4`) rather than a webcam — swap the `cv2.VideoCapture(...)` line for `cv2.VideoCapture(0)` to use a live camera. Adjust these constants at the top of the script for your setup:

- `CONF` — detection confidence threshold (default `0.35`)
- `ROI` — the `(x1, y1, x2, y2)` region people are counted inside

Press `Esc` to quit.

### Legacy line-crossing counter

```bash
python counter.py
```

Reads from a video file by default (`test_singleperson.mp4`) — swap in `cv2.VideoCapture(0)` for a webcam. Counts people crossing horizontal lines partway up and down the frame and prints running totals to the console. Press `Esc` to quit.

## Roadmap

- [ ] Web front end for live count display
- [ ] Configurable ROI/line coordinates via CLI args or config file instead of hardcoded values
- [ ] Persistent logging of counts over time

## Credits

Built on top of [epcm18/PeopleCounting-ComputerVision](https://github.com/epcm18/PeopleCounting-ComputerVision).
