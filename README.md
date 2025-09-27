# Clothing Object Detection

This repository contains a lightweight command-line utility for running clothing object detection on a single image. The script wraps a custom-trained [Ultralytics YOLOv8](https://docs.ultralytics.com) model and saves an annotated image and (optionally) a CSV file describing every detected item.

## Features

- Runs inference with any YOLOv8 detection model that has been trained to recognise clothing classes.
- Saves an annotated copy of the input image with bounding boxes and class labels.
- Emits a CSV report containing the bounding box coordinates and confidence score of each detection.
- Customisable confidence threshold, image size and compute device.

## Getting started

1. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\\Scripts\\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Obtain YOLO weights that are trained for clothing detection**

   The repository does not ship with pretrained weights. You can either fine-tune YOLOv8 on your own fashion dataset or download community models that are already trained for apparel. A few options:

   - Train on the [Fashionpedia](https://fashionpedia.github.io/home/) or [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) datasets using the [`ultralytics` CLI](https://docs.ultralytics.com/modes/train/).
   - Download a community model such as the [Roboflow Universe "Clothing Detection" YOLOv8 weights](https://universe.roboflow.com/roboflow-universe-projects/clothing-detection) and place the exported `.pt` file in a local `weights/` folder.

   After downloading or training a model, make sure the weights file is accessible to the script (e.g. `weights/yolov8n-fashion.pt`).

## Usage

```bash
python src/detect_clothes.py path/to/image.jpg \
    --model weights/yolov8n-fashion.pt \
    --confidence 0.35 \
    --output outputs/annotated.jpg \
    --save-csv outputs/detections.csv
```

### Arguments

| Flag | Description |
|------|-------------|
| `image` | Required path to the image on which to run detection. |
| `--model` | Path to the YOLOv8 weights file (.pt) trained to detect clothing (default: `weights/yolov8n-fashion.pt`). |
| `--confidence` | Minimum confidence required for a detection to be kept (default: `0.25`). |
| `--device` | Compute device override. Examples: `cpu`, `cuda`, `cuda:0`. Defaults to the behaviour provided by Ultralytics. |
| `--img-size` | Inference image size. Must match what the model expects (default: `640`). |
| `--output` | Optional path for the annotated image. Defaults to `<input_stem>_detections.jpg` in the source directory. |
| `--save-csv` | Optional path for the CSV report summarising detections. |

The command prints the location of the annotated image (and CSV, if requested). If no `--output` is provided, the annotated file is written alongside the source image using the `<image>_detections.jpg` naming convention.

## Tips

- Lower the `--confidence` threshold if you need to capture smaller or partially occluded garments; increase it to reduce false positives.
- When using a GPU-enabled environment, pass `--device cuda` for faster inference.
- The CSV file contains one row per detection with the bounding-box coordinates in pixel space (`xmin`, `ymin`, `xmax`, `ymax`). These can be used for downstream analytics or integrations.

## License

This project is provided as-is under the MIT license. Refer to the license of any third-party model or dataset you use for training to ensure compatibility.
