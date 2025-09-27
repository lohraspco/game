"""Segment clothing items in an image and highlight the topmost garment.

The script uses colour clustering (k-means) followed by contour extraction to
obtain candidate garment masks. It then ranks those masks by the smallest
vertical pixel coordinate, which approximates the "topmost" garment when the
camera looks down the z-axis (top-down view). The best ranked mask is rendered
on top of the original image and saved to disk.

Example
-------
    python cloth_segmentation.py \
        --image /path/to/pile.jpg \
        --segments 5 \
        --min-area 800 \
        --output out.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np


@dataclass
class GarmentMask:
    """Container describing a single garment instance."""

    mask: np.ndarray
    contour: np.ndarray

    @property
    def top_edge(self) -> int:
        """Return the minimum y coordinate covered by the mask."""

        ys, _ = np.where(self.mask > 0)
        return int(ys.min()) if ys.size else np.iinfo(np.int32).max

    @property
    def area(self) -> float:
        return float(cv2.countNonZero(self.mask))

    @property
    def bounding_box(self) -> Tuple[int, int, int, int]:
        x, y, w, h = cv2.boundingRect(self.contour)
        return x, y, w, h


def _cluster_image(image: np.ndarray, segments: int) -> np.ndarray:
    """Perform k-means clustering on the image pixels."""

    h, w, c = image.shape
    data = image.reshape((-1, c)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, _ = cv2.kmeans(
        data,
        segments,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )
    return labels.reshape((h, w))


def _refine_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened


def _extract_garments(label_map: np.ndarray, min_area: int) -> List[GarmentMask]:
    garments: List[GarmentMask] = []
    unique_labels = np.unique(label_map)
    for label in unique_labels:
        mask = np.zeros_like(label_map, dtype=np.uint8)
        mask[label_map == label] = 255
        refined = _refine_mask(mask)
        contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            component_mask = np.zeros_like(mask)
            cv2.drawContours(component_mask, [contour], -1, 255, thickness=cv2.FILLED)
            garments.append(GarmentMask(component_mask, contour))
    return garments


def _select_topmost(garments: Iterable[GarmentMask]) -> GarmentMask | None:
    garments = list(garments)
    if not garments:
        return None
    garments.sort(key=lambda g: (g.top_edge, -g.area))
    return garments[0]


def _render_overlay(image: np.ndarray, garment: GarmentMask) -> np.ndarray:
    overlay = image.copy()
    mask_rgb = cv2.cvtColor(garment.mask, cv2.COLOR_GRAY2BGR)
    colour = np.array([0, 255, 0], dtype=np.uint8)
    overlay = np.where(mask_rgb > 0, (overlay * 0.4 + colour * 0.6).astype(np.uint8), overlay)
    x, y, w, h = garment.bounding_box
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
    cv2.putText(
        overlay,
        "Topmost garment",
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def process_image(image_path: Path, segments: int, min_area: int, output_path: Path) -> Path:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    blurred = cv2.pyrMeanShiftFiltering(image, 21, 51)
    label_map = _cluster_image(blurred, segments)
    garments = _extract_garments(label_map, min_area=min_area)
    topmost = _select_topmost(garments)
    if topmost is None:
        raise RuntimeError("No garment-sized segments were detected. Try lowering --min-area or increasing --segments.")

    overlay = _render_overlay(image, topmost)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment and highlight the topmost garment in an image.")
    parser.add_argument("--image", required=True, type=Path, help="Path to the input image.")
    parser.add_argument(
        "--segments",
        type=int,
        default=4,
        help="Number of colour clusters to generate. Increase for more complex piles.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=1000,
        help="Minimum contour area (in pixels) required to consider a cluster a garment.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("topmost_garment.png"),
        help="Path to save the visualised result.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_path = process_image(args.image, args.segments, args.min_area, args.output)
    print(f"Saved overlay with topmost garment highlighted to {result_path}")


if __name__ == "__main__":
    main()

