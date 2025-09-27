## Cloth pile segmentation

This repository contains a small utility script, `cloth_segmentation.py`, that
segments a top-down image of a pile of clothes, extracts the individual
garments and highlights the one that is spatially "topmost" in the frame.

The script approximates the topmost garment as the segment whose mask reaches
the smallest vertical (y) coordinate. This assumption is reasonable for images
captured from the z-axis because the garment on the top of the pile will be the
one that extends the furthest upwards in the photo. If this assumption does not
hold for your images you can adjust the ranking logic in `_select_topmost`.

### Requirements

Install the dependencies in a Python environment (3.9+ recommended):

```
pip install opencv-python numpy
```

### Usage

```
python cloth_segmentation.py \
    --image path/to/your/image.jpg \
    --segments 5 \
    --min-area 800 \
    --output topmost.png
```

* `--segments` controls the number of colour clusters produced by the k-means
  step. Increase it if the garments are visually similar.
* `--min-area` filters out small blobs that are unlikely to be clothes.

The output image highlights the topmost garment in green and annotates it with a
bounding box. If no garment-sized segment is detected the script raises an
error with hints on how to adjust the parameters.

