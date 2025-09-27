"""Command-line utility for running clothing object detection on images."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

import cv2
import numpy as np
from ultralytics import YOLO


NamesType = Sequence[str] | Mapping[int, str]


@dataclass
class Detection:
    """Container that stores a single detection result."""

    label: str
    confidence: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @classmethod
    def from_result(cls, names: NamesType, row: Sequence[float]) -> "Detection":
        class_id = int(row[5])
        if isinstance(names, Mapping):
            label = names.get(class_id, str(class_id))
        else:
            label = names[class_id] if 0 <= class_id < len(names) else str(class_id)
        confidence = float(row[4])
        xmin, ymin, xmax, ymax = map(float, row[:4])
        return cls(label=label, confidence=confidence, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)


@dataclass
class DetectionSummary:
    """Summary of detections for a single image."""

    source: Path
    detections: List[Detection]
    annotated_image: np.ndarray
    default_output_path: Path

    def to_rows(self) -> Iterable[Sequence[str]]:
        """Convert detections into a set of CSV rows."""

        for det in self.detections:
            yield (
                str(self.source),
                det.label,
                f"{det.confidence:.3f}",
                f"{det.xmin:.2f}",
                f"{det.ymin:.2f}",
                f"{det.xmax:.2f}",
                f"{det.ymax:.2f}",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect clothing items in an image using a YOLO model.")
    parser.add_argument("image", type=Path, help="Path to the input image")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("weights/yolov8n-fashion.pt"),
        help="Path to the trained YOLO model weights (.pt file)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold for filtering detections (default: 0.25)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run inference on (e.g. 'cpu', 'cuda', 'cuda:0'). Defaults to auto selection.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Image size that the YOLO model will resize to before inference (default: 640)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path where the annotated image will be saved. Defaults to <image>_detections.jpg",
    )
    parser.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="Optional CSV path where detection metadata will be stored",
    )
    return parser.parse_args()


def run_inference(
    image_path: Path,
    model_path: Path,
    confidence: float = 0.25,
    device: str | None = None,
    img_size: int = 640,
) -> DetectionSummary:
    """Run the YOLO model on the provided image and return a summary of the detections."""

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            "YOLO weights not found. Download or train a clothing detection model and point --model to it."
        )

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(image_path),
        conf=confidence,
        imgsz=img_size,
        device=device or "",
    )

    if not results:
        raise RuntimeError("Model did not return any results. Ensure the weights file is valid.")

    result = results[0]
    names: NamesType = result.names

    detections: List[Detection] = []
    if getattr(result, "boxes", None) is not None and getattr(result.boxes, "data", None) is not None:
        data = result.boxes.data.cpu().numpy()
        detections = [Detection.from_result(names, row) for row in data]

    annotated_image = result.plot()
    default_output_path = image_path.with_name(f"{image_path.stem}_detections.jpg")

    return DetectionSummary(
        source=image_path,
        detections=detections,
        annotated_image=annotated_image,
        default_output_path=default_output_path,
    )


def save_csv(summary: DetectionSummary, csv_path: Path) -> None:
    """Persist detections to disk as a CSV file."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "label", "confidence", "xmin", "ymin", "xmax", "ymax"])
        writer.writerows(summary.to_rows())


def main() -> None:
    args = parse_args()

    summary = run_inference(
        image_path=args.image,
        model_path=args.model,
        confidence=args.confidence,
        device=args.device,
        img_size=args.img_size,
    )

    output_path = args.output or summary.default_output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved = cv2.imwrite(str(output_path), summary.annotated_image)
    if not saved:
        raise RuntimeError(f"Failed to save annotated image to {output_path}")
    print(f"Annotated image saved to: {output_path}")

    if not summary.detections:
        print("No clothing items were detected with the current confidence threshold.")
    else:
        print("Detections:")
        for det in summary.detections:
            print(
                f"  - {det.label} | confidence={det.confidence:.2%} | "
                f"box=({det.xmin:.1f}, {det.ymin:.1f}, {det.xmax:.1f}, {det.ymax:.1f})"
            )

    if args.save_csv:
        save_csv(summary, args.save_csv)
        print(f"Detections saved to: {args.save_csv}")


if __name__ == "__main__":
    main()
