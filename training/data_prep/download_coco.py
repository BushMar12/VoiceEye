"""COCO 2017 fetcher + annotation→YOLO converter.

The fetcher is a thin wrapper around three public zip URLs. The converter
(coco_ann_to_yolo / convert_coco_split) is pure logic and unit-tested.
"""

from __future__ import annotations
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Optional

import requests

from training.data_prep.taxonomy import COCO_KEEP_MAP

COCO_URLS = {
    'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
    'val_images':   'http://images.cocodataset.org/zips/val2017.zip',
    'annotations':  'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
}


def coco_ann_to_yolo(ann: dict, img_w: int, img_h: int) -> Optional[str]:
    """Convert a single COCO annotation dict to a YOLO label line.

    Returns None if the annotation should be dropped (class not kept, crowd
    mask, or zero-area bbox).
    """
    if ann.get('iscrowd', 0) == 1:
        return None
    cls = COCO_KEEP_MAP.get(ann['category_id'])
    if cls is None:
        return None
    x, y, w, h = ann['bbox']
    if w <= 0 or h <= 0:
        return None
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return f"{cls} {cx:.8f} {cy:.8f} {nw:.8f} {nh:.8f}"


def convert_coco_split(ann_file: Path, out_labels_dir: Path, prefix: str = 'coco') -> None:
    """Read a COCO annotation JSON and write YOLO .txt files to out_labels_dir.

    One .txt per image (even if empty — negative samples are valuable).
    Filenames: {prefix}_{original_filename_stem}.txt
    """
    data = json.loads(Path(ann_file).read_text())
    imgs = {img['id']: img for img in data['images']}
    lines_per_img: dict[int, list[str]] = {img_id: [] for img_id in imgs}
    for ann in data['annotations']:
        img = imgs.get(ann['image_id'])
        if img is None:
            continue
        line = coco_ann_to_yolo(ann, img['width'], img['height'])
        if line is not None:
            lines_per_img[ann['image_id']].append(line)

    out_labels_dir.mkdir(parents=True, exist_ok=True)
    for img_id, img in imgs.items():
        stem = Path(img['file_name']).stem
        out = out_labels_dir / f"{prefix}_{stem}.txt"
        out.write_text('\n'.join(lines_per_img[img_id]))


def _download_with_retry(url: str, dest: Path, attempts: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return  # idempotent
    tmp = dest.with_suffix(dest.suffix + '.part')
    import time
    delay = 1
    last_err = None
    for i in range(attempts):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
            tmp.rename(dest)
            return
        except Exception as e:
            last_err = e
            if tmp.exists():
                tmp.unlink()
            if i < attempts - 1:
                time.sleep(delay)
                delay *= 4
    raise RuntimeError(f"Failed to download {url} after {attempts} attempts: {last_err}")


def download_and_convert_coco(raw_dir: Path, processed_labels_dir: Path, processed_images_dir: Path) -> None:
    """Fetch COCO 2017, unzip, then convert annotations + copy images."""
    raw_dir = Path(raw_dir)
    for name, url in COCO_URLS.items():
        dest = raw_dir / f"{name}.zip"
        print(f"[coco] download {name}")
        _download_with_retry(url, dest)
        extract_dir = raw_dir / name
        if not extract_dir.exists():
            with zipfile.ZipFile(dest) as z:
                z.extractall(extract_dir)

    ann_dir = raw_dir / 'annotations' / 'annotations'
    for split, ann_name in [('train', 'instances_train2017.json'),
                            ('val',   'instances_val2017.json')]:
        print(f"[coco] convert {split}")
        convert_coco_split(ann_dir / ann_name, processed_labels_dir, prefix='coco')

    # Copy images: train2017/ and val2017/ → processed_images_dir/ with coco_ prefix
    processed_images_dir.mkdir(parents=True, exist_ok=True)
    for split_zip_name, subdir in [('train_images', 'train2017'), ('val_images', 'val2017')]:
        src = raw_dir / split_zip_name / subdir
        for img in src.glob('*.jpg'):
            dst = processed_images_dir / f"coco_{img.stem}.jpg"
            if not dst.exists():
                shutil.copy2(img, dst)


if __name__ == '__main__':
    import sys
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/raw/coco')
    lbl = Path('data/processed/labels')
    img = Path('data/processed/images')
    download_and_convert_coco(raw, lbl, img)
