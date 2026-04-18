"""Open Images v7 fetcher + row→YOLO converter + per-class cap."""

from __future__ import annotations
import csv
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Optional

from training.data_prep.taxonomy import OIV7_CLASS_MAP


def oiv7_row_to_yolo(row: dict) -> Optional[str]:
    """Convert one OIv7 boxes CSV row to a YOLO label line.

    Returns None for unknown class, group-of bbox, or degenerate bbox.
    Row fields: LabelName, XMin, XMax, YMin, YMax, IsGroupOf (0/1).
    """
    if int(row.get('IsGroupOf', 0)) == 1:
        return None
    cls = OIV7_CLASS_MAP.get(row['LabelName'])
    if cls is None:
        return None
    x0, x1 = float(row['XMin']), float(row['XMax'])
    y0, y1 = float(row['YMin']), float(row['YMax'])
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    return f"{cls} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}"


def cap_instances_per_class(rows: list[dict], cap: int) -> list[dict]:
    """Keep up to `cap` instances per LabelName. Images are kept as a unit.

    Iterates images in insertion order; adds all boxes for an image to the
    output until adding them would exceed the cap for some class.
    """
    # Group rows by image
    by_image: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_image[r['ImageID']].append(r)

    counts_per_class: dict[str, int] = defaultdict(int)
    kept: list[dict] = []
    for img_id, img_rows in by_image.items():
        # What new counts would this image produce?
        img_counts: dict[str, int] = defaultdict(int)
        for r in img_rows:
            img_counts[r['LabelName']] += 1
        # Reject if any class would exceed cap
        if any(counts_per_class[c] + n > cap for c, n in img_counts.items()):
            continue
        kept.extend(img_rows)
        for c, n in img_counts.items():
            counts_per_class[c] += n
    return kept


def _read_boxes_csv(path: Path) -> list[dict]:
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def download_and_convert_oiv7(raw_dir: Path, processed_labels_dir: Path,
                              processed_images_dir: Path, cap: int = 3000) -> None:
    """Fetch OIv7 subset (Door/Stairs/Pole) and convert to YOLO labels.

    Uses the `oi_download_dataset` CLI from the openimages PyPI package.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    classes = list(OIV7_CLASS_MAP.keys())  # ['Door', 'Stairs', 'Pole']

    # Skip download if CSV already present
    boxes_csv = raw_dir / 'train-annotations-bbox.csv'
    if not boxes_csv.exists():
        cmd = [
            'oi_download_dataset',
            '--base_dir', str(raw_dir),
            '--labels', *classes,
            '--format', 'pascal',   # we only need the images + boxes CSV
            '--csv_dir', str(raw_dir),
        ]
        print(f"[oiv7] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    rows = _read_boxes_csv(boxes_csv)
    # Pre-filter to our three classes before capping
    rows = [r for r in rows if r['LabelName'] in OIV7_CLASS_MAP]
    rows = cap_instances_per_class(rows, cap=cap)

    # Group kept rows by ImageID, write one label file per image
    by_image: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        line = oiv7_row_to_yolo(r)
        if line is not None:
            by_image[r['ImageID']].append(line)

    processed_labels_dir.mkdir(parents=True, exist_ok=True)
    processed_images_dir.mkdir(parents=True, exist_ok=True)
    for img_id, lines in by_image.items():
        (processed_labels_dir / f"oiv7_{img_id}.txt").write_text('\n'.join(lines))
        # Copy image (CLI downloads to raw_dir/train/<ImageID>.jpg)
        for candidate in [raw_dir / 'train' / f"{img_id}.jpg",
                          raw_dir / 'validation' / f"{img_id}.jpg",
                          raw_dir / 'test' / f"{img_id}.jpg"]:
            if candidate.exists():
                dst = processed_images_dir / f"oiv7_{img_id}.jpg"
                if not dst.exists():
                    shutil.copy2(candidate, dst)
                break


if __name__ == '__main__':
    download_and_convert_oiv7(
        Path('data/raw/oiv7'),
        Path('data/processed/labels'),
        Path('data/processed/images'),
    )
