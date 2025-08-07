# Video Inference with MindYOLO and MindSpore

This repository contains a video inference application built using the [MindSpore](https://www.mindspore.cn/en) framework and the [MindYOLO](https://github.com/mindspore-lab/mindyolo) model zoo. It performs real-time object detection and identity tracking on video files using a lightweight YOLOv8n model and a custom-built tracking system.

## 📌 Project Overview

The system is designed as a modular video processing pipeline composed of:

- **Object Detection**: Uses a pretrained YOLOv8n model to identify objects in each frame.
- **Identity Tracking**: Maintains consistent object IDs across frames using a normalized-distance proximity algorithm.
- **Visualization Engine**: Overlays bounding boxes, unique IDs, motion trails, and counters on the video output.

> ⚠️ This project is a proof-of-concept aimed at learning the MindSpore ecosystem. It is not production-grade and can benefit from future optimizations and improved tracking algorithms.

## 💽 Example Use Case

The model has been tested on various scenarios including:
- Pet movement (e.g. dogs)
- Street traffic from surveillance footage
- Crowded pedestrian areas

While it performs well in simple environments, it may struggle in occluded or densely packed scenes due to its basic tracking mechanism.

## 📦 Installation Guide

### 🔧 Prerequisites

Before you begin, make sure you have the following installed:

- Python ≥ 3.8
- [MindSpore](https://www.mindspore.cn/en) (with appropriate backend: CPU, GPU, or Ascend)
- `pip` or `conda` for package management

### 🐍 Recommended Installation Steps

```bash
# (Optional) Create a virtual environment
python -m venv mindyolo-env
source mindyolo-env/bin/activate  # On Windows: mindyolo-env\Scripts\activate

# Clone the repository
git clone https://github.com/your-username/video-inference-mindyolo.git
cd video-inference-mindyolo

# Install dependencies
pip install -r requirements.txt
```

> 💡 Be sure to install MindSpore manually according to your hardware:  
> [MindSpore Installation Guide](https://www.mindspore.cn/install)

## ▶️ Usage

To run the inference:

```bash
python video_infer.py --config ./configs/yolov8/yolov8n.yaml \
--video_path path/to/input.mp4 \
--output_path path/to/output.mp4 \
--weight ./path/to/yolov8n.ckpt \
--device_target CPU
```

### 🔍 Supported Command-Line Options

| Category       | Option             | Description |
|----------------|--------------------|-------------|
| Input/Output   | `--video_path`     | Input video file (required) |
|                | `--output_path`    | Path to save output (default: `./runs_infer`) |
|                | `--fps`            | Frame rate for output video |
| Processing     | `--frame_skip`     | Process 1 frame every N frames (0 = all) |
|                | `--img_size`       | Image size for inference (default: 640) |
|                | `--conf_thres`     | Detection confidence threshold |
| Tracking       | `--max_trajectory` | Number of trajectory points shown (0 = disabled) |
|                | `--min_confidence` | Minimum confidence to show detection |
| Hardware       | `--device_target`  | Backend: `CPU`, `GPU`, or `Ascend` |

## 📂 Folder Structure

```
.
├── configs/                  # YOLOv8 model configs
├── output/                   # Processed videos
├── weights/                  # Model checkpoints
├── video_infer.py            # Main inference script
└── README.md
```

## 📊 Future Work

- Integrate advanced tracking (e.g. DeepSORT, ByteTrack)
- Instance segmentation support
- Web interface for real-time demos
- Fine-tuned training for specific domains (e.g. aerial, night scenes)

## 📚 References

- [MindYOLO GitHub](https://github.com/mindspore-lab/mindyolo)
- [MindSpore Official Website](https://www.mindspore.cn/en)
- [COCO Dataset](https://cocodataset.org/#home)

## 👤 Author

Samuele Orsenigo  
Artificial Intelligence Project — Politecnico di Milano

---

**License:** MIT *(or your preferred license — remember to add a `LICENSE` file)*  
**Repository Link:** [Insert your GitHub repo link here]

