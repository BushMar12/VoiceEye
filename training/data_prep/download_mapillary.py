"""Mapillary Vistas Research fetcher + polygon→bbox converter.

Mapillary Vistas ships as per-image JSON with polygon segmentation annotations.
We convert polygons to tight axis-aligned bboxes for our YOLO format.

Requires MAPILLARY_TOKEN env var (Research Edition signup gate). If absent,
the fetcher prints the signup URL and skips — validate_dataset.py will then
fail the class-coverage hard gate, which is the desired loud failure.
"""

from __future__ import annotations
import json
import os
import shutil
from pathlib import Path
from typing import Optional

from training.data_prep.taxonomy import MAPILLARY_CLASS_MAP

MAPILLARY_SIGNUP_URL = 'https://www.mapillary.com/dataset/vistas'
MIN_BBOX_AREA_FRAC = 0.001  # bbox smaller than this fraction of image → drop


def polygon_to_bbox(poly: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    """Tight axis-aligned bbox over polygon vertices. Returns (x0, y0, x1, y1)."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_annotation_to_yolo(ann: dict, img_w: int, img_h: int) -> Optional[str]:
    """Convert one polygon annotation to a YOLO label line. None if dropped."""
    cls = MAPILLARY_CLASS_MAP.get(ann['label'])
    if cls is None:
        return None
    x0, y0, x1, y1 = polygon_to_bbox(ann['polygon'])
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None
    # Normalize
    nw = w / img_w
    nh = h / img_h
    if nw * nh < MIN_BBOX_AREA_FRAC:
        return None
    cx = (x0 + x1) / 2 / img_w
    cy = (y0 + y1) / 2 / img_h
    return f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def _load_vistas_json(path: Path) -> tuple[list[dict], int, int]:
    """Load Mapillary Vistas per-image JSON. Returns (annotations, width, height).

    Vistas JSON schema (v2.0):
      { "width": W, "height": H, "objects": [
          {"label": "...", "polygon": [[x,y], ...]}, ... ] }
    """
    data = json.loads(Path(path).read_text())
    anns = [{'label': o['label'], 'polygon': [tuple(p) for p in o['polygon']]}
            for o in data['objects']]
    return anns, data['width'], data['height']


def download_and_convert_mapillary(raw_dir: Path, processed_labels_dir: Path,
                                   processed_images_dir: Path) -> bool:
    """Fetch Mapillary Vistas Research subset. Returns False if skipped.

    The actual Mapillary download API requires an account + signed URL
    per-session — fetching is manual. This function expects the user to have
    unpacked the Vistas archive under raw_dir already:

        raw_dir/training/images/*.jpg
        raw_dir/training/v2.0/polygons/*.json
        raw_dir/validation/images/*.jpg
        raw_dir/validation/v2.0/polygons/*.json
    """
    if not os.environ.get('MAPILLARY_TOKEN'):
        print(f"[mapillary] MAPILLARY_TOKEN not set. Skipping.")
        print(f"[mapillary] Sign up for Research Edition: {MAPILLARY_SIGNUP_URL}")
        print(f"[mapillary] Then unpack under {raw_dir} and set MAPILLARY_TOKEN=1.")
        return False

    raw_dir = Path(raw_dir)
    if not (raw_dir / 'training').exists():
        print(f"[mapillary] raw dataset not found under {raw_dir}. Skipping.")
        return False

    processed_labels_dir.mkdir(parents=True, exist_ok=True)
    processed_images_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for split in ['training', 'validation']:
        poly_dir = raw_dir / split / 'v2.0' / 'polygons'
        img_dir = raw_dir / split / 'images'
        if not poly_dir.exists():
            continue
        for json_file in poly_dir.glob('*.json'):
            anns, w, h = _load_vistas_json(json_file)
            lines = []
            for ann in anns:
                line = polygon_annotation_to_yolo(ann, w, h)
                if line is not None:
                    lines.append(line)
            if not lines:
                continue  # images with zero kept classes add no value here
            stem = json_file.stem
            (processed_labels_dir / f"mapillary_{stem}.txt").write_text('\n'.join(lines))
            src_img = img_dir / f"{stem}.jpg"
            if src_img.exists():
                dst = processed_images_dir / f"mapillary_{stem}.jpg"
                if not dst.exists():
                    shutil.copy2(src_img, dst)
            converted += 1
    print(f"[mapillary] converted {converted} images")
    return True


if __name__ == '__main__':
    download_and_convert_mapillary(
        Path('data/raw/mapillary'),
        Path('data/processed/labels'),
        Path('data/processed/images'),
    )
