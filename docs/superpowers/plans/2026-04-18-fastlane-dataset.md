# Fast Lane Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `training/data_prep/` — a reproducible pipeline that fetches COCO 2017 + Open Images v7 + Mapillary Vistas Research, converts them to a unified 48-class YOLO-format dataset, validates it, and uploads to ClearML as `voiceeye-fastlane-v1`.

**Architecture:** Ten single-responsibility Python modules under `training/data_prep/`, orchestrated by `make_dataset.py`. Pure conversion logic (JSON/CSV/polygon → YOLO bbox, stratified split, weight computation, validation) is TDD'd with pytest. Network fetchers are thin wrappers around source CLIs/HTTP and tested manually.

**Tech Stack:** Python 3.10+, pytest, PyYAML, requests, clearml (Dataset API), numpy, Pillow (PIL — for image sampling / overlay rendering).

**Reference spec:** `docs/superpowers/specs/2026-04-18-fastlane-dataset-design.md`

---

## File structure

```
training/data_prep/
├── __init__.py
├── taxonomy.py                 # 48-class index + per-source class maps
├── download_coco.py            # Fetch COCO 2017 + coco_ann_to_yolo()
├── download_open_images.py     # OIv7 fetch + oiv7_row_to_yolo() + cap logic
├── download_mapillary.py       # Mapillary fetch + polygon_to_bbox()
├── merge_to_yolo.py            # Stitch sources, write source_manifest.csv
├── make_splits.py              # stratified_split() → data/final/ + data.yaml
├── compute_class_weights.py    # inverse_frequency_weights()
├── validate_dataset.py         # Hard + soft gates, sample overlays, report
├── upload_to_clearml.py        # Dataset.create() → finalize()
├── make_dataset.py             # Orchestrator + disk check
└── tests/
    ├── __init__.py
    ├── test_taxonomy.py
    ├── test_coco_convert.py
    ├── test_oiv7_convert.py
    ├── test_mapillary_convert.py
    ├── test_merge.py
    ├── test_splits.py
    ├── test_weights.py
    └── test_validate.py
```

All paths below are relative to the repo root `C:\Users\Bush\OneDrive\OneDrive - UTS\VoiceEye`.

---

## Task 1: Scaffold package + pytest + taxonomy

**Files:**
- Create: `training/data_prep/__init__.py` (empty)
- Create: `training/data_prep/tests/__init__.py` (empty)
- Create: `training/data_prep/taxonomy.py`
- Create: `training/data_prep/tests/test_taxonomy.py`
- Modify: `training/requirements.txt`
- Create: `training/data_prep/pytest.ini`

- [ ] **Step 1: Add test dependencies to `training/requirements.txt`**

Append two lines:

```
pytest>=8.0.0
Pillow>=10.0.0
requests>=2.31.0
```

- [ ] **Step 2: Create empty `__init__.py` files**

```bash
touch training/data_prep/__init__.py
mkdir -p training/data_prep/tests
touch training/data_prep/tests/__init__.py
```

- [ ] **Step 3: Create `training/data_prep/pytest.ini`**

```ini
[pytest]
testpaths = training/data_prep/tests
python_files = test_*.py
```

- [ ] **Step 4: Write the failing test at `training/data_prep/tests/test_taxonomy.py`**

```python
from training.data_prep.taxonomy import (
    CLASSES,
    COCO_KEEP_MAP,
    OIV7_CLASS_MAP,
    MAPILLARY_CLASS_MAP,
)


def test_classes_has_48_entries():
    assert len(CLASSES) == 48


def test_classes_are_unique():
    assert len(set(CLASSES)) == 48


def test_six_new_classes_present():
    for new_cls in ['stairs', 'curb', 'crosswalk', 'pole', 'bollard', 'door']:
        assert new_cls in CLASSES


def test_dropped_coco_classes_absent():
    for dropped in ['airplane', 'train', 'boat', 'banana', 'toothbrush', 'hair drier']:
        assert dropped not in CLASSES


def test_coco_keep_map_covers_42_classes():
    # Keys are COCO category_ids (1..90 with gaps); values are our class indices
    assert len(COCO_KEEP_MAP) == 42
    for our_idx in COCO_KEEP_MAP.values():
        assert 0 <= our_idx < 48


def test_oiv7_map_has_three_classes():
    # Keys are OIv7 class names; values are our indices
    assert set(OIV7_CLASS_MAP.keys()) == {'Door', 'Stairs', 'Pole'}
    for our_idx in OIV7_CLASS_MAP.values():
        assert 0 <= our_idx < 48


def test_mapillary_map_has_three_classes():
    assert set(MAPILLARY_CLASS_MAP.keys()) == {
        'crosswalk-plain', 'object--support--pole-bollard', 'curb'
    } or set(MAPILLARY_CLASS_MAP.keys()) == {
        'construction--flat--crosswalk-plain',
        'object--support--pole',
        'construction--flat--curb',
    }
    # Spec leaves the exact Mapillary class key string to whatever the dataset
    # ships; this test just pins the count and that values are valid.
    assert len(MAPILLARY_CLASS_MAP) == 3


def test_new_classes_not_in_source_maps():
    # The 3 new classes from OIv7 and 3 from Mapillary cover all 6 new ones
    new_indices = {CLASSES.index(c) for c in
                   ['stairs', 'curb', 'crosswalk', 'pole', 'bollard', 'door']}
    mapped_new = set(OIV7_CLASS_MAP.values()) | set(MAPILLARY_CLASS_MAP.values())
    assert new_indices == mapped_new
```

- [ ] **Step 5: Run the test to verify it fails**

```bash
cd "C:/Users/Bush/OneDrive/OneDrive - UTS/VoiceEye"
python -m pytest training/data_prep/tests/test_taxonomy.py -v
```

Expected: ImportError — `taxonomy` module does not exist.

- [ ] **Step 6: Create `training/data_prep/taxonomy.py`**

```python
"""48-class taxonomy and source→target class maps for VoiceEye Fast Lane."""

# Final class order. Index = class id written to YOLO label files.
CLASSES = [
    # 42 kept from COCO (preserving a sensible order, not COCO's)
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
    'traffic light', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog',
    'backpack', 'umbrella', 'handbag', 'suitcase',
    'bottle', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'keyboard', 'cell phone',
    'microwave', 'oven', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors',
    # 6 new navigation-critical classes
    'stairs', 'curb', 'crosswalk', 'pole', 'bollard', 'door',
]
assert len(CLASSES) == 48, f"CLASSES must have 48 entries, got {len(CLASSES)}"

# Keys are COCO 2017 category_ids (gappy, max 90). Values are our CLASSES indices.
# Dropped: airplane(5), train(7), boat(9), fire hydrant(11), horse(19), sheep(20),
# cow(21), elephant(22), bear(23), zebra(24), giraffe(25), tie(32), frisbee(34),
# skis(35), snowboard(36), sports ball(37), kite(38), baseball bat(39),
# baseball glove(40), skateboard(41), surfboard(42), tennis racket(43),
# wine glass(46), banana(52), apple(53), sandwich(54), orange(55), broccoli(56),
# carrot(57), hot dog(58), pizza(59), donut(60), cake(61), remote(75),
# toaster(80), teddy bear(88), hair drier(89), toothbrush(90).
_COCO_NAME_TO_ID = {
    'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'bus': 6, 'truck': 8,
    'traffic light': 10, 'stop sign': 13, 'parking meter': 14, 'bench': 15,
    'bird': 16, 'cat': 17, 'dog': 18,
    'backpack': 27, 'umbrella': 28, 'handbag': 31, 'suitcase': 33,
    'bottle': 44, 'cup': 47, 'fork': 48, 'knife': 49, 'spoon': 50, 'bowl': 51,
    'chair': 62, 'couch': 63, 'potted plant': 64, 'bed': 65,
    'dining table': 67, 'toilet': 70,
    'tv': 72, 'laptop': 73, 'mouse': 74, 'keyboard': 76, 'cell phone': 77,
    'microwave': 78, 'oven': 79, 'sink': 81, 'refrigerator': 82,
    'book': 84, 'clock': 85, 'vase': 86, 'scissors': 87,
}
COCO_KEEP_MAP = {
    coco_id: CLASSES.index(name) for name, coco_id in _COCO_NAME_TO_ID.items()
}
assert len(COCO_KEEP_MAP) == 42

# Open Images v7 uses human-readable class names in the boxes CSV.
OIV7_CLASS_MAP = {
    'Door':   CLASSES.index('door'),
    'Stairs': CLASSES.index('stairs'),
    'Pole':   CLASSES.index('pole'),
}

# Mapillary Vistas v2.0 research class names (verify at pipeline run time —
# the Mapillary downloader prints a sample list of class strings if a key
# does not match).
MAPILLARY_CLASS_MAP = {
    'construction--flat--crosswalk-plain': CLASSES.index('crosswalk'),
    'object--support--pole':               CLASSES.index('bollard'),
    'construction--flat--curb':            CLASSES.index('curb'),
}
```

> Note: Mapillary's `object--support--pole` is actually used for bollards in
> Vistas' class hierarchy; `pole-utility` is a different leaf. If the dataset
> version in use exposes a dedicated `bollard` key, `download_mapillary.py`
> prints the available class list and the engineer updates this map once.

- [ ] **Step 7: Run the test to verify it passes**

```bash
python -m pytest training/data_prep/tests/test_taxonomy.py -v
```

Expected: 7 passed.

- [ ] **Step 8: Commit**

```bash
git add training/data_prep/ training/requirements.txt
git commit -m "feat(data_prep): scaffold package + 48-class taxonomy

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: COCO annotation → YOLO converter (TDD)

**Files:**
- Create: `training/data_prep/download_coco.py`
- Create: `training/data_prep/tests/test_coco_convert.py`

- [ ] **Step 1: Write the failing test at `training/data_prep/tests/test_coco_convert.py`**

```python
import pytest
from training.data_prep.download_coco import coco_ann_to_yolo, convert_coco_split


def test_kept_class_converts_to_yolo():
    # COCO 'person' (category_id=1), image 800x600, bbox=[100, 200, 50, 80]
    ann = {'category_id': 1, 'bbox': [100, 200, 50, 80], 'iscrowd': 0}
    img_w, img_h = 800, 600
    line = coco_ann_to_yolo(ann, img_w, img_h)
    # Our 'person' is class 0
    parts = line.split()
    assert parts[0] == '0'
    # cx = (100 + 50/2) / 800 = 0.15625; cy = (200 + 80/2) / 600 = 0.4
    assert float(parts[1]) == pytest.approx(0.15625)
    assert float(parts[2]) == pytest.approx(0.4)
    assert float(parts[3]) == pytest.approx(50 / 800)
    assert float(parts[4]) == pytest.approx(80 / 600)


def test_dropped_class_returns_none():
    # COCO 'airplane' (category_id=5) — dropped
    ann = {'category_id': 5, 'bbox': [0, 0, 10, 10], 'iscrowd': 0}
    assert coco_ann_to_yolo(ann, 100, 100) is None


def test_iscrowd_returns_none():
    ann = {'category_id': 1, 'bbox': [0, 0, 10, 10], 'iscrowd': 1}
    assert coco_ann_to_yolo(ann, 100, 100) is None


def test_zero_width_bbox_returns_none():
    ann = {'category_id': 1, 'bbox': [0, 0, 0, 10], 'iscrowd': 0}
    assert coco_ann_to_yolo(ann, 100, 100) is None


def test_convert_coco_split_writes_one_txt_per_image(tmp_path):
    ann_json = {
        'images': [
            {'id': 1, 'file_name': '000000000001.jpg', 'width': 100, 'height': 100},
            {'id': 2, 'file_name': '000000000002.jpg', 'width': 200, 'height': 200},
        ],
        'annotations': [
            {'image_id': 1, 'category_id': 1, 'bbox': [10, 10, 20, 20], 'iscrowd': 0},
            {'image_id': 2, 'category_id': 5, 'bbox': [0, 0, 10, 10], 'iscrowd': 0},
        ],
    }
    import json
    ann_file = tmp_path / 'ann.json'
    ann_file.write_text(json.dumps(ann_json))
    out_dir = tmp_path / 'labels'
    out_dir.mkdir()

    convert_coco_split(ann_file, out_dir, prefix='coco')

    # Image 1 has a person kept
    assert (out_dir / 'coco_000000000001.txt').read_text().strip() != ''
    # Image 2 has only airplane dropped → file still created (negative sample) but empty
    assert (out_dir / 'coco_000000000002.txt').exists()
    assert (out_dir / 'coco_000000000002.txt').read_text() == ''
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest training/data_prep/tests/test_coco_convert.py -v
```

Expected: ImportError — `download_coco` does not exist.

- [ ] **Step 3: Create `training/data_prep/download_coco.py`**

```python
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
    return f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest training/data_prep/tests/test_coco_convert.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add training/data_prep/download_coco.py training/data_prep/tests/test_coco_convert.py
git commit -m "feat(data_prep): COCO 2017 fetcher + ann→YOLO converter

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: Open Images v7 converter + per-class cap (TDD)

**Files:**
- Create: `training/data_prep/download_open_images.py`
- Create: `training/data_prep/tests/test_oiv7_convert.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from training.data_prep.download_open_images import (
    oiv7_row_to_yolo,
    cap_instances_per_class,
)


def test_door_row_converts():
    row = {'LabelName': 'Door', 'XMin': 0.1, 'XMax': 0.3, 'YMin': 0.2, 'YMax': 0.4,
           'IsGroupOf': 0}
    line = oiv7_row_to_yolo(row)
    parts = line.split()
    # 'door' is class 47 in CLASSES (last entry)
    assert parts[0] == '47'
    assert float(parts[1]) == pytest.approx(0.2)   # cx = (0.1+0.3)/2
    assert float(parts[2]) == pytest.approx(0.3)   # cy = (0.2+0.4)/2
    assert float(parts[3]) == pytest.approx(0.2)   # w  = 0.3-0.1
    assert float(parts[4]) == pytest.approx(0.2)   # h  = 0.4-0.2


def test_unknown_class_returns_none():
    row = {'LabelName': 'Banana', 'XMin': 0, 'XMax': 0.1, 'YMin': 0, 'YMax': 0.1,
           'IsGroupOf': 0}
    assert oiv7_row_to_yolo(row) is None


def test_group_of_returns_none():
    row = {'LabelName': 'Door', 'XMin': 0, 'XMax': 0.1, 'YMin': 0, 'YMax': 0.1,
           'IsGroupOf': 1}
    assert oiv7_row_to_yolo(row) is None


def test_degenerate_bbox_returns_none():
    row = {'LabelName': 'Door', 'XMin': 0.5, 'XMax': 0.5, 'YMin': 0, 'YMax': 0.1,
           'IsGroupOf': 0}
    assert oiv7_row_to_yolo(row) is None


def test_cap_instances_per_class_caps_correctly():
    # 10 doors across 5 images (2 per image), 5 stairs across 3 images
    rows = []
    for i in range(5):
        for _ in range(2):
            rows.append({'ImageID': f'img_door_{i}', 'LabelName': 'Door',
                         'XMin': 0, 'XMax': 0.1, 'YMin': 0, 'YMax': 0.1,
                         'IsGroupOf': 0})
    for i in range(3):
        rows.append({'ImageID': f'img_stairs_{i}', 'LabelName': 'Stairs',
                     'XMin': 0, 'XMax': 0.1, 'YMin': 0, 'YMax': 0.1,
                     'IsGroupOf': 0})
    # Cap doors at 6 instances → should keep 3 images (2 doors each)
    capped = cap_instances_per_class(rows, cap=6)
    door_rows = [r for r in capped if r['LabelName'] == 'Door']
    stair_rows = [r for r in capped if r['LabelName'] == 'Stairs']
    assert len(door_rows) == 6
    assert len(stair_rows) == 3  # under cap, all kept
    # Images are kept as a unit — 3 door images with 2 boxes each
    assert len({r['ImageID'] for r in door_rows}) == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest training/data_prep/tests/test_oiv7_convert.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create `training/data_prep/download_open_images.py`**

```python
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
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


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
```

- [ ] **Step 4: Add `openimages` to requirements**

Append one line to `training/requirements.txt`:

```
openimages>=0.2.0
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest training/data_prep/tests/test_oiv7_convert.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add training/data_prep/download_open_images.py training/data_prep/tests/test_oiv7_convert.py training/requirements.txt
git commit -m "feat(data_prep): Open Images v7 converter + per-class cap

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: Mapillary polygon → bbox converter (TDD)

**Files:**
- Create: `training/data_prep/download_mapillary.py`
- Create: `training/data_prep/tests/test_mapillary_convert.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from training.data_prep.download_mapillary import (
    polygon_to_bbox,
    polygon_annotation_to_yolo,
)


def test_polygon_to_bbox_basic():
    # Square polygon at pixel coords
    poly = [(10, 20), (30, 20), (30, 40), (10, 40)]
    x0, y0, x1, y1 = polygon_to_bbox(poly)
    assert (x0, y0, x1, y1) == (10, 20, 30, 40)


def test_polygon_to_bbox_irregular():
    poly = [(5, 5), (15, 10), (12, 25), (3, 20)]
    x0, y0, x1, y1 = polygon_to_bbox(poly)
    assert (x0, y0, x1, y1) == (3, 5, 15, 25)


def test_polygon_annotation_mapped_class():
    ann = {
        'label': 'construction--flat--crosswalk-plain',
        'polygon': [(100, 200), (300, 200), (300, 280), (100, 280)],
    }
    line = polygon_annotation_to_yolo(ann, img_w=1000, img_h=800)
    parts = line.split()
    # 'crosswalk' is class 44 in CLASSES
    assert parts[0] == '44'
    # bbox: (100,200,300,280) → cx=0.2, cy=0.3, w=0.2, h=0.1
    assert float(parts[1]) == pytest.approx(0.2)
    assert float(parts[2]) == pytest.approx(0.3)
    assert float(parts[3]) == pytest.approx(0.2)
    assert float(parts[4]) == pytest.approx(0.1)


def test_polygon_annotation_unknown_class_returns_none():
    ann = {'label': 'sky', 'polygon': [(0, 0), (10, 0), (10, 10), (0, 10)]}
    assert polygon_annotation_to_yolo(ann, 100, 100) is None


def test_polygon_annotation_tiny_bbox_returns_none():
    # Area = 1/10000 = 0.0001 < 0.001 threshold
    ann = {
        'label': 'construction--flat--crosswalk-plain',
        'polygon': [(0, 0), (1, 0), (1, 1), (0, 1)],
    }
    assert polygon_annotation_to_yolo(ann, 100, 100) is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest training/data_prep/tests/test_mapillary_convert.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create `training/data_prep/download_mapillary.py`**

```python
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
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest training/data_prep/tests/test_mapillary_convert.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add training/data_prep/download_mapillary.py training/data_prep/tests/test_mapillary_convert.py
git commit -m "feat(data_prep): Mapillary polygon→bbox converter

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: Merge → source manifest (TDD)

**Files:**
- Create: `training/data_prep/merge_to_yolo.py`
- Create: `training/data_prep/tests/test_merge.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
from training.data_prep.merge_to_yolo import write_source_manifest


def test_source_manifest_groups_by_prefix(tmp_path):
    images = tmp_path / 'images'
    images.mkdir()
    (images / 'coco_001.jpg').write_bytes(b'x')
    (images / 'oiv7_abc.jpg').write_bytes(b'x')
    (images / 'mapillary_zzz.jpg').write_bytes(b'x')

    manifest = tmp_path / 'manifest.csv'
    write_source_manifest(images, manifest)

    text = manifest.read_text()
    lines = text.strip().split('\n')
    assert lines[0] == 'filename,source,original_id'
    rows = sorted(lines[1:])
    assert rows == sorted([
        'coco_001.jpg,coco,001',
        'mapillary_zzz.jpg,mapillary,zzz',
        'oiv7_abc.jpg,oiv7,abc',
    ])


def test_unprefixed_filename_raises(tmp_path):
    images = tmp_path / 'images'
    images.mkdir()
    (images / 'mystery.jpg').write_bytes(b'x')
    import pytest
    with pytest.raises(ValueError, match='unknown source'):
        write_source_manifest(images, tmp_path / 'm.csv')
```

- [ ] **Step 2: Run test — expect ImportError**

```bash
python -m pytest training/data_prep/tests/test_merge.py -v
```

- [ ] **Step 3: Create `training/data_prep/merge_to_yolo.py`**

```python
"""Source manifest builder. Merging itself is done in-place by each
download_*.py (they all write to the same processed/images + processed/labels
directories), so `merge_to_yolo.py` is really just the manifest step."""

from __future__ import annotations
import csv
from pathlib import Path

SOURCES = ('coco', 'oiv7', 'mapillary')


def _parse_prefixed(filename: str) -> tuple[str, str]:
    for src in SOURCES:
        prefix = f"{src}_"
        if filename.startswith(prefix):
            return src, filename[len(prefix):].rsplit('.', 1)[0]
    raise ValueError(f"unknown source prefix for {filename!r}")


def write_source_manifest(images_dir: Path, manifest_path: Path) -> None:
    images_dir = Path(images_dir)
    rows = []
    for img in sorted(images_dir.iterdir()):
        if not img.is_file():
            continue
        source, orig_id = _parse_prefixed(img.name)
        rows.append({'filename': img.name, 'source': source, 'original_id': orig_id})
    with open(manifest_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['filename', 'source', 'original_id'])
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    write_source_manifest(
        Path('data/processed/images'),
        Path('data/processed/source_manifest.csv'),
    )
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest training/data_prep/tests/test_merge.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add training/data_prep/merge_to_yolo.py training/data_prep/tests/test_merge.py
git commit -m "feat(data_prep): source manifest builder

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 6: Source-stratified 90/10 split + data.yaml (TDD)

**Files:**
- Create: `training/data_prep/make_splits.py`
- Create: `training/data_prep/tests/test_splits.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
from training.data_prep.make_splits import stratified_split, build_data_yaml


def test_stratified_split_ratio_per_source():
    pool = (
        [('coco', f'coco_{i}.jpg') for i in range(100)] +
        [('oiv7', f'oiv7_{i}.jpg') for i in range(50)] +
        [('mapillary', f'mapillary_{i}.jpg') for i in range(20)]
    )
    train, val = stratified_split(pool, val_frac=0.1, seed=42)
    # 90/10 per source
    assert len([t for t in train if t.startswith('coco_')]) == 90
    assert len([v for v in val if v.startswith('coco_')]) == 10
    assert len([t for t in train if t.startswith('oiv7_')]) == 45
    assert len([v for v in val if v.startswith('oiv7_')]) == 5
    assert len([t for t in train if t.startswith('mapillary_')]) == 18
    assert len([v for v in val if v.startswith('mapillary_')]) == 2


def test_stratified_split_disjoint():
    pool = [('coco', f'coco_{i}.jpg') for i in range(100)]
    train, val = stratified_split(pool, val_frac=0.1, seed=42)
    assert set(train).isdisjoint(set(val))
    assert set(train) | set(val) == {p[1] for p in pool}


def test_stratified_split_deterministic():
    pool = [('coco', f'coco_{i}.jpg') for i in range(100)]
    a_train, a_val = stratified_split(pool, val_frac=0.1, seed=42)
    b_train, b_val = stratified_split(pool, val_frac=0.1, seed=42)
    assert a_train == b_train
    assert a_val == b_val


def test_build_data_yaml(tmp_path):
    out = tmp_path / 'data.yaml'
    build_data_yaml(out)
    text = out.read_text()
    assert 'nc: 48' in text
    assert 'train: images/train' in text
    assert 'val: images/val' in text
    assert 'person' in text
    assert 'door' in text
```

- [ ] **Step 2: Run test — expect ImportError**

```bash
python -m pytest training/data_prep/tests/test_splits.py -v
```

- [ ] **Step 3: Create `training/data_prep/make_splits.py`**

```python
"""Source-stratified train/val split + data.yaml writer."""

from __future__ import annotations
import csv
import random
import shutil
from pathlib import Path

import yaml

from training.data_prep.taxonomy import CLASSES


def stratified_split(pool: list[tuple[str, str]],
                     val_frac: float = 0.1,
                     seed: int = 42) -> tuple[list[str], list[str]]:
    """Split `pool` [(source, filename), ...] into (train, val) filenames.

    Each source is shuffled independently with a per-source deterministic seed
    derived from `seed`, then the last `val_frac` goes to val.
    """
    from collections import defaultdict
    by_source: dict[str, list[str]] = defaultdict(list)
    for src, fname in pool:
        by_source[src].append(fname)

    train, val = [], []
    for src in sorted(by_source):  # deterministic source order
        items = sorted(by_source[src])  # deterministic starting order
        rng = random.Random(seed)
        rng.shuffle(items)
        cut = int(len(items) * (1 - val_frac))
        train.extend(items[:cut])
        val.extend(items[cut:])
    return train, val


def build_data_yaml(out_path: Path) -> None:
    data = {
        'path': '.',
        'train': 'images/train',
        'val':   'images/val',
        'nc': len(CLASSES),
        'names': list(CLASSES),
    }
    with open(out_path, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def materialize_splits(manifest_path: Path,
                       processed_dir: Path,
                       final_dir: Path,
                       val_frac: float = 0.1,
                       seed: int = 42) -> tuple[int, int]:
    """Read source manifest, split, and copy images+labels to final_dir.

    Returns (n_train, n_val).
    """
    with open(manifest_path, newline='') as f:
        rows = list(csv.DictReader(f))
    pool = [(r['source'], r['filename']) for r in rows]
    train, val = stratified_split(pool, val_frac=val_frac, seed=seed)

    for split_name, names in [('train', train), ('val', val)]:
        img_out = final_dir / 'images' / split_name
        lbl_out = final_dir / 'labels' / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for fname in names:
            stem = Path(fname).stem
            shutil.copy2(processed_dir / 'images' / fname, img_out / fname)
            src_lbl = processed_dir / 'labels' / f"{stem}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, lbl_out / f"{stem}.txt")
            else:
                (lbl_out / f"{stem}.txt").write_text('')  # negative sample

    build_data_yaml(final_dir / 'data.yaml')
    return len(train), len(val)


if __name__ == '__main__':
    materialize_splits(
        Path('data/processed/source_manifest.csv'),
        Path('data/processed'),
        Path('data/final'),
    )
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest training/data_prep/tests/test_splits.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add training/data_prep/make_splits.py training/data_prep/tests/test_splits.py
git commit -m "feat(data_prep): source-stratified split + data.yaml

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 7: Class weights (TDD)

**Files:**
- Create: `training/data_prep/compute_class_weights.py`
- Create: `training/data_prep/tests/test_weights.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from training.data_prep.compute_class_weights import (
    inverse_frequency_weights,
    count_instances_per_class,
)


def test_equal_counts_gives_equal_weights():
    counts = {i: 100 for i in range(48)}
    w = inverse_frequency_weights(counts, nc=48)
    for v in w.values():
        assert v == pytest.approx(1.0)


def test_rare_class_gets_higher_weight():
    counts = {i: 1000 for i in range(48)}
    counts[0] = 10  # class 0 is 100x rarer
    w = inverse_frequency_weights(counts, nc=48)
    assert w[0] > w[1]
    # Mean should be 1.0
    assert sum(w.values()) / 48 == pytest.approx(1.0)


def test_zero_count_does_not_divide_by_zero():
    counts = {i: 100 for i in range(48)}
    counts[5] = 0
    w = inverse_frequency_weights(counts, nc=48)
    # Weight clamped: treated as count=1 internally
    assert w[5] > w[0]  # class 5 rarest
    assert all(v > 0 for v in w.values())


def test_count_instances_per_class(tmp_path):
    # Two label files: 3 class-0, 1 class-2, 0 elsewhere
    (tmp_path / 'a.txt').write_text('0 0.5 0.5 0.1 0.1\n0 0.3 0.3 0.1 0.1')
    (tmp_path / 'b.txt').write_text('0 0.2 0.2 0.1 0.1\n2 0.7 0.7 0.2 0.2')
    counts = count_instances_per_class(tmp_path, nc=48)
    assert counts[0] == 3
    assert counts[2] == 1
    assert counts[1] == 0
```

- [ ] **Step 2: Run test — expect ImportError**

```bash
python -m pytest training/data_prep/tests/test_weights.py -v
```

- [ ] **Step 3: Create `training/data_prep/compute_class_weights.py`**

```python
"""Inverse-frequency class weights (mean-1 normalized)."""

from __future__ import annotations
from pathlib import Path
import yaml

from training.data_prep.taxonomy import CLASSES


def count_instances_per_class(labels_dir: Path, nc: int) -> dict[int, int]:
    counts = {i: 0 for i in range(nc)}
    for txt in Path(labels_dir).glob('*.txt'):
        for line in txt.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            cls = int(line.split()[0])
            if 0 <= cls < nc:
                counts[cls] += 1
    return counts


def inverse_frequency_weights(counts: dict[int, int], nc: int) -> dict[int, float]:
    inv = {c: 1.0 / max(counts.get(c, 0), 1) for c in range(nc)}
    total = sum(inv.values())
    # Mean-1 normalization: scale so sum(weights) == nc (mean 1.0)
    return {c: (inv[c] / total) * nc for c in range(nc)}


def write_class_weights(labels_train_dir: Path, out_path: Path) -> None:
    counts = count_instances_per_class(labels_train_dir, nc=len(CLASSES))
    weights = inverse_frequency_weights(counts, nc=len(CLASSES))
    data = {
        'nc': len(CLASSES),
        'classes': [
            {'id': c, 'name': CLASSES[c], 'count': counts[c], 'weight': round(weights[c], 6)}
            for c in range(len(CLASSES))
        ],
    }
    Path(out_path).write_text(yaml.safe_dump(data, sort_keys=False))


if __name__ == '__main__':
    write_class_weights(
        Path('data/final/labels/train'),
        Path('data/final/class_weights.yaml'),
    )
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest training/data_prep/tests/test_weights.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add training/data_prep/compute_class_weights.py training/data_prep/tests/test_weights.py
git commit -m "feat(data_prep): inverse-frequency class weights

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 8: Validate dataset — hard gates (TDD)

**Files:**
- Create: `training/data_prep/validate_dataset.py`
- Create: `training/data_prep/tests/test_validate.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from pathlib import Path
from training.data_prep.validate_dataset import (
    check_image_label_pairing,
    check_bbox_sanity,
    check_split_disjointness,
    check_class_coverage,
    ValidationError,
)


def test_pairing_passes_when_matched(tmp_path):
    (tmp_path / 'images').mkdir()
    (tmp_path / 'labels').mkdir()
    (tmp_path / 'images' / 'a.jpg').write_bytes(b'x')
    (tmp_path / 'labels' / 'a.txt').write_text('')
    check_image_label_pairing(tmp_path / 'images', tmp_path / 'labels')


def test_pairing_fails_on_orphan_image(tmp_path):
    (tmp_path / 'images').mkdir()
    (tmp_path / 'labels').mkdir()
    (tmp_path / 'images' / 'a.jpg').write_bytes(b'x')
    (tmp_path / 'images' / 'b.jpg').write_bytes(b'x')
    (tmp_path / 'labels' / 'a.txt').write_text('')
    with pytest.raises(ValidationError, match='orphan'):
        check_image_label_pairing(tmp_path / 'images', tmp_path / 'labels')


def test_bbox_sanity_passes_on_valid(tmp_path):
    (tmp_path / 'a.txt').write_text('0 0.5 0.5 0.1 0.1')
    check_bbox_sanity(tmp_path, nc=48)


def test_bbox_sanity_fails_on_out_of_range(tmp_path):
    (tmp_path / 'a.txt').write_text('0 1.5 0.5 0.1 0.1')
    with pytest.raises(ValidationError, match='out of range'):
        check_bbox_sanity(tmp_path, nc=48)


def test_bbox_sanity_fails_on_zero_area(tmp_path):
    (tmp_path / 'a.txt').write_text('0 0.5 0.5 0 0.1')
    with pytest.raises(ValidationError, match='zero'):
        check_bbox_sanity(tmp_path, nc=48)


def test_bbox_sanity_fails_on_bad_class_id(tmp_path):
    (tmp_path / 'a.txt').write_text('99 0.5 0.5 0.1 0.1')
    with pytest.raises(ValidationError, match='class id'):
        check_bbox_sanity(tmp_path, nc=48)


def test_split_disjointness_passes(tmp_path):
    (tmp_path / 'train').mkdir()
    (tmp_path / 'val').mkdir()
    (tmp_path / 'train' / 'a.jpg').write_bytes(b'x')
    (tmp_path / 'val' / 'b.jpg').write_bytes(b'x')
    check_split_disjointness(tmp_path / 'train', tmp_path / 'val')


def test_split_disjointness_fails_on_overlap(tmp_path):
    (tmp_path / 'train').mkdir()
    (tmp_path / 'val').mkdir()
    (tmp_path / 'train' / 'a.jpg').write_bytes(b'x')
    (tmp_path / 'val' / 'a.jpg').write_bytes(b'x')
    with pytest.raises(ValidationError, match='both train and val'):
        check_split_disjointness(tmp_path / 'train', tmp_path / 'val')


def test_class_coverage_passes(tmp_path):
    # All 48 classes present
    lines = [f"{c} 0.5 0.5 0.1 0.1" for c in range(48)]
    (tmp_path / 'a.txt').write_text('\n'.join(lines))
    check_class_coverage(tmp_path, nc=48)


def test_class_coverage_fails_on_missing(tmp_path):
    (tmp_path / 'a.txt').write_text('0 0.5 0.5 0.1 0.1')
    with pytest.raises(ValidationError, match='missing classes'):
        check_class_coverage(tmp_path, nc=48)
```

- [ ] **Step 2: Run test — expect ImportError**

```bash
python -m pytest training/data_prep/tests/test_validate.py -v
```

- [ ] **Step 3: Create `training/data_prep/validate_dataset.py` (hard gates only — soft gates added in Task 9)**

```python
"""Dataset validation. Hard gates raise ValidationError; soft gates warn."""

from __future__ import annotations
from pathlib import Path


class ValidationError(Exception):
    pass


def _image_stems(images_dir: Path) -> set[str]:
    return {p.stem for p in Path(images_dir).glob('*') if p.is_file()}


def _label_stems(labels_dir: Path) -> set[str]:
    return {p.stem for p in Path(labels_dir).glob('*.txt')}


def check_image_label_pairing(images_dir: Path, labels_dir: Path) -> None:
    imgs = _image_stems(images_dir)
    lbls = _label_stems(labels_dir)
    only_imgs = imgs - lbls
    only_lbls = lbls - imgs
    if only_imgs or only_lbls:
        raise ValidationError(
            f"orphan: {len(only_imgs)} images without labels, "
            f"{len(only_lbls)} labels without images"
        )


def check_bbox_sanity(labels_dir: Path, nc: int) -> None:
    for txt in Path(labels_dir).glob('*.txt'):
        for lineno, line in enumerate(txt.read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValidationError(f"{txt}:{lineno} — expected 5 fields, got {len(parts)}")
            try:
                cls = int(parts[0])
                cx, cy, w, h = (float(x) for x in parts[1:])
            except ValueError:
                raise ValidationError(f"{txt}:{lineno} — non-numeric field")
            if cls < 0 or cls >= nc:
                raise ValidationError(f"{txt}:{lineno} — class id {cls} not in [0,{nc})")
            for name, v in [('cx', cx), ('cy', cy), ('w', w), ('h', h)]:
                if v < 0 or v > 1:
                    raise ValidationError(f"{txt}:{lineno} — {name}={v} out of range [0,1]")
            if w <= 0 or h <= 0:
                raise ValidationError(f"{txt}:{lineno} — zero or negative w/h")


def check_split_disjointness(train_images_dir: Path, val_images_dir: Path) -> None:
    train = {p.name for p in Path(train_images_dir).glob('*') if p.is_file()}
    val = {p.name for p in Path(val_images_dir).glob('*') if p.is_file()}
    overlap = train & val
    if overlap:
        raise ValidationError(
            f"{len(overlap)} filenames appear in both train and val: "
            f"{sorted(overlap)[:3]}..."
        )


def check_class_coverage(labels_train_dir: Path, nc: int) -> None:
    present: set[int] = set()
    for txt in Path(labels_train_dir).glob('*.txt'):
        for line in txt.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            present.add(int(line.split()[0]))
    missing = set(range(nc)) - present
    if missing:
        raise ValidationError(
            f"missing classes in train: {sorted(missing)} "
            f"(extinct class would divide-by-zero in weight calc)"
        )


def run_hard_gates(final_dir: Path, nc: int = 48) -> None:
    final_dir = Path(final_dir)
    for split in ('train', 'val'):
        check_image_label_pairing(final_dir / 'images' / split, final_dir / 'labels' / split)
        check_bbox_sanity(final_dir / 'labels' / split, nc=nc)
    check_split_disjointness(final_dir / 'images' / 'train', final_dir / 'images' / 'val')
    check_class_coverage(final_dir / 'labels' / 'train', nc=nc)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest training/data_prep/tests/test_validate.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add training/data_prep/validate_dataset.py training/data_prep/tests/test_validate.py
git commit -m "feat(data_prep): hard validation gates

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 9: Validate dataset — soft gates + report + sample overlays

**Files:**
- Modify: `training/data_prep/validate_dataset.py` (add functions)
- Modify: `training/data_prep/tests/test_validate.py` (add 3 tests)

- [ ] **Step 1: Append failing tests to `training/data_prep/tests/test_validate.py`**

```python
# Append at end of test_validate.py:
import json
from training.data_prep.validate_dataset import (
    build_report,
    render_samples,
)


def test_build_report_counts_per_class(tmp_path):
    (tmp_path / 'labels' / 'train').mkdir(parents=True)
    (tmp_path / 'labels' / 'val').mkdir(parents=True)
    (tmp_path / 'labels' / 'train' / 'coco_a.txt').write_text('0 0.5 0.5 0.1 0.1\n2 0.5 0.5 0.1 0.1')
    (tmp_path / 'labels' / 'val' / 'oiv7_b.txt').write_text('47 0.5 0.5 0.1 0.1')
    report = build_report(tmp_path, nc=48)
    assert report['train']['per_class'][0] == 1
    assert report['train']['per_class'][2] == 1
    assert report['val']['per_class'][47] == 1
    assert report['train']['per_source']['coco'] == 1
    assert report['val']['per_source']['oiv7'] == 1


def test_render_samples_writes_20_files(tmp_path):
    # Create 25 dummy images + labels
    from PIL import Image
    (tmp_path / 'images').mkdir()
    (tmp_path / 'labels').mkdir()
    for i in range(25):
        Image.new('RGB', (100, 100), 'black').save(tmp_path / 'images' / f"coco_{i}.jpg")
        (tmp_path / 'labels' / f"coco_{i}.txt").write_text('0 0.5 0.5 0.3 0.3')
    out = tmp_path / 'samples'
    render_samples(tmp_path / 'images', tmp_path / 'labels', out, n=20, seed=42)
    assert len(list(out.glob('*.jpg'))) == 20


def test_warn_under_balanced_class(capsys, tmp_path):
    from training.data_prep.validate_dataset import warn_class_balance
    counts = {i: 500 for i in range(48)}
    counts[3] = 50  # under 100 threshold
    warn_class_balance(counts)
    captured = capsys.readouterr()
    assert 'WARN' in captured.out
    assert 'motorcycle' in captured.out  # class 3 is 'motorcycle'
```

- [ ] **Step 2: Run tests — expect failures (functions not defined)**

```bash
python -m pytest training/data_prep/tests/test_validate.py -v
```

- [ ] **Step 3: Append to `training/data_prep/validate_dataset.py`**

```python
# Append to validate_dataset.py:

import json
import random as _random

from training.data_prep.taxonomy import CLASSES


def build_report(final_dir: Path, nc: int) -> dict:
    """Count instances per class and per source for train + val."""
    from collections import defaultdict
    report: dict = {}
    for split in ('train', 'val'):
        per_class = [0] * nc
        per_source: dict[str, int] = defaultdict(int)
        for txt in (Path(final_dir) / 'labels' / split).glob('*.txt'):
            for src in ('coco', 'oiv7', 'mapillary'):
                if txt.name.startswith(src + '_'):
                    per_source[src] += 1
                    break
            for line in txt.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                cls = int(line.split()[0])
                if 0 <= cls < nc:
                    per_class[cls] += 1
        report[split] = {'per_class': per_class, 'per_source': dict(per_source)}
    return report


def render_samples(images_dir: Path, labels_dir: Path, out_dir: Path,
                   n: int = 20, seed: int = 42) -> None:
    """Render n random images with bbox overlays for eyeball QA."""
    from PIL import Image, ImageDraw
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted(Path(images_dir).glob('*'))
    rng = _random.Random(seed)
    rng.shuffle(imgs)
    for img_path in imgs[:n]:
        im = Image.open(img_path).convert('RGB')
        W, H = im.size
        draw = ImageDraw.Draw(im)
        lbl = labels_dir / f"{img_path.stem}.txt"
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                cls, cx, cy, w, h = line.split()
                cls = int(cls); cx, cy, w, h = map(float, (cx, cy, w, h))
                x0 = (cx - w / 2) * W
                y0 = (cy - h / 2) * H
                x1 = (cx + w / 2) * W
                y1 = (cy + h / 2) * H
                draw.rectangle([x0, y0, x1, y1], outline='red', width=2)
                draw.text((x0 + 2, y0 + 2), CLASSES[cls], fill='red')
        im.save(out_dir / img_path.name)


def warn_class_balance(counts: dict[int, int], threshold_train: int = 100) -> None:
    under = [(c, counts[c]) for c in counts if counts[c] < threshold_train]
    for c, n in under:
        print(f"WARN: class {c} ({CLASSES[c]}) has only {n} instances in train")


def warn_val_coverage(per_class_val: list[int], threshold_val: int = 10) -> None:
    for c, n in enumerate(per_class_val):
        if n < threshold_val:
            print(f"WARN: class {c} ({CLASSES[c]}) has only {n} instances in val "
                  f"— mAP will be noisy")


def run_soft_gates(final_dir: Path, nc: int = 48) -> dict:
    """Run soft gates. Returns the report dict. Never raises."""
    final_dir = Path(final_dir)
    report = build_report(final_dir, nc=nc)
    counts_train = {i: report['train']['per_class'][i] for i in range(nc)}
    warn_class_balance(counts_train)
    warn_val_coverage(report['val']['per_class'])

    qa_dir = Path('data/qa')
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / 'report.json').write_text(json.dumps(report, indent=2))
    render_samples(
        final_dir / 'images' / 'train',
        final_dir / 'labels' / 'train',
        qa_dir / 'samples',
        n=20,
    )
    return report


def validate_all(final_dir: Path, nc: int = 48) -> dict:
    run_hard_gates(final_dir, nc=nc)
    return run_soft_gates(final_dir, nc=nc)


if __name__ == '__main__':
    validate_all(Path('data/final'))
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest training/data_prep/tests/test_validate.py -v
```

Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add training/data_prep/validate_dataset.py training/data_prep/tests/test_validate.py
git commit -m "feat(data_prep): soft gates, report, sample overlays

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 10: ClearML upload

**Files:**
- Create: `training/data_prep/upload_to_clearml.py`

(No unit test — this is a thin wrapper around ClearML's Dataset API; manually verified by running and checking the printed dataset_id.)

- [ ] **Step 1: Create `training/data_prep/upload_to_clearml.py`**

```python
"""Upload data/final/ to ClearML as voiceeye-fastlane-v1."""

from __future__ import annotations
import json
from pathlib import Path

from clearml import Dataset


REQUIRED_LICENSES = ('coco.txt', 'oiv7.txt', 'mapillary.txt')


def upload(final_dir: Path, name: str = 'voiceeye-fastlane-v1',
           project: str = 'VoiceEye') -> str:
    final_dir = Path(final_dir)

    license_dir = final_dir / 'LICENSES'
    missing = [f for f in REQUIRED_LICENSES if not (license_dir / f).exists()]
    if missing:
        raise RuntimeError(
            f"Missing license files: {missing}. Refusing to upload — "
            f"attribution is required by the source dataset terms."
        )

    ds = Dataset.create(dataset_name=name, dataset_project=project)
    ds.add_files(path=str(final_dir))

    # Add manifest with provenance
    manifest = {
        'name': name,
        'sources': ['COCO 2017', 'Open Images v7', 'Mapillary Vistas Research'],
        'nc': 48,
        'split': {'train_frac': 0.9, 'val_frac': 0.1, 'seed': 42},
        'license': 'non-commercial (Mapillary Research Edition)',
    }
    manifest_path = final_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))

    ds.upload()
    ds.finalize()
    print(f"[clearml] dataset_id = {ds.id}")
    print(f"[clearml] Set `dataset_id: \"{ds.id}\"` in training/config.yaml")
    return ds.id


if __name__ == '__main__':
    upload(Path('data/final'))
```

- [ ] **Step 2: Commit**

```bash
git add training/data_prep/upload_to_clearml.py
git commit -m "feat(data_prep): ClearML Dataset upload with license guard

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 11: Orchestrator `make_dataset.py` + disk guard

**Files:**
- Create: `training/data_prep/make_dataset.py`

- [ ] **Step 1: Create `training/data_prep/make_dataset.py`**

```python
"""Run the full dataset pipeline end-to-end.

    python training/data_prep/make_dataset.py
"""

from __future__ import annotations
import shutil
from pathlib import Path

from training.data_prep.download_coco import download_and_convert_coco
from training.data_prep.download_open_images import download_and_convert_oiv7
from training.data_prep.download_mapillary import download_and_convert_mapillary
from training.data_prep.merge_to_yolo import write_source_manifest
from training.data_prep.make_splits import materialize_splits
from training.data_prep.compute_class_weights import write_class_weights
from training.data_prep.validate_dataset import validate_all
from training.data_prep.upload_to_clearml import upload


MIN_FREE_GB = 40


def _check_disk(path: Path) -> None:
    free_gb = shutil.disk_usage(str(path.resolve().anchor)).free / (1024 ** 3)
    if free_gb < MIN_FREE_GB:
        raise RuntimeError(
            f"Only {free_gb:.1f} GB free — need >= {MIN_FREE_GB} GB for raw downloads."
        )


def _copy_licenses(final_dir: Path) -> None:
    licenses_src = Path('training/data_prep/licenses')
    licenses_dst = final_dir / 'LICENSES'
    licenses_dst.mkdir(parents=True, exist_ok=True)
    for name in ('coco.txt', 'oiv7.txt', 'mapillary.txt'):
        src = licenses_src / name
        dst = licenses_dst / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def main(upload_to_clearml: bool = True) -> None:
    data = Path('data')
    _check_disk(data)

    raw = data / 'raw'
    processed = data / 'processed'
    final = data / 'final'

    # 1-3: Downloads + conversions
    download_and_convert_coco(raw / 'coco', processed / 'labels', processed / 'images')
    download_and_convert_oiv7(raw / 'oiv7', processed / 'labels', processed / 'images')
    download_and_convert_mapillary(raw / 'mapillary', processed / 'labels', processed / 'images')

    # 4: Manifest
    manifest_path = processed / 'source_manifest.csv'
    write_source_manifest(processed / 'images', manifest_path)

    # 5: Splits + data.yaml
    n_train, n_val = materialize_splits(manifest_path, processed, final)
    print(f"[split] train={n_train} val={n_val}")

    # 6: Class weights
    write_class_weights(final / 'labels' / 'train', final / 'class_weights.yaml')

    # 7: Licenses
    _copy_licenses(final)

    # 8: Validate (raises on hard gate failure)
    report = validate_all(final)
    print(f"[validate] train per-source counts: {report['train']['per_source']}")
    print(f"[validate] val   per-source counts: {report['val']['per_source']}")

    # 9: Upload
    if upload_to_clearml:
        upload(final)


if __name__ == '__main__':
    import sys
    skip_upload = '--no-upload' in sys.argv
    main(upload_to_clearml=not skip_upload)
```

- [ ] **Step 2: Create `training/data_prep/licenses/` directory with placeholder license text files**

```bash
mkdir -p training/data_prep/licenses
```

Create `training/data_prep/licenses/coco.txt`:

```
COCO 2017 is released under the Creative Commons Attribution 4.0 License.
See https://cocodataset.org/#termsofuse

Citation:
Lin et al., "Microsoft COCO: Common Objects in Context", ECCV 2014.
```

Create `training/data_prep/licenses/oiv7.txt`:

```
Open Images Dataset V7 annotations are licensed by Google LLC under
CC BY 4.0 license. The images are listed as having a CC BY 2.0 license.
See https://storage.googleapis.com/openimages/web/factsfigures_v7.html

Citation:
Kuznetsova et al., "The Open Images Dataset V4", IJCV 2020.
```

Create `training/data_prep/licenses/mapillary.txt`:

```
Mapillary Vistas Dataset — Research Edition.
Licensed for non-commercial research use only.
See https://www.mapillary.com/dataset/vistas for terms.

Citation:
Neuhold et al., "The Mapillary Vistas Dataset for Semantic Understanding
of Street Scenes", ICCV 2017.

DO NOT redistribute images or train commercial models on this data.
```

- [ ] **Step 3: Smoke-test the imports** (no network, no downloads)

```bash
python -c "from training.data_prep.make_dataset import main; print('imports ok')"
```

Expected: `imports ok`.

- [ ] **Step 4: Commit**

```bash
git add training/data_prep/make_dataset.py training/data_prep/licenses/
git commit -m "feat(data_prep): orchestrator + license files

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 12: Full test suite + docs

**Files:**
- Create: `training/data_prep/README.md`

- [ ] **Step 1: Run the full data_prep test suite**

```bash
python -m pytest training/data_prep/tests/ -v
```

Expected: all tests pass (sum of tasks 1, 2, 3, 4, 5, 6, 7, 8, 9 = 7+5+5+5+2+4+4+10+3 = 45 passed).

- [ ] **Step 2: Create `training/data_prep/README.md`**

```markdown
# Fast Lane Dataset Builder

Builds `voiceeye-fastlane-v1` — a 48-class YOLO dataset for VoiceEye's
Fast Lane, merged from three public sources.

See the [design spec](../../docs/superpowers/specs/2026-04-18-fastlane-dataset-design.md)
for background and rationale.

## Quickstart

```bash
# 1. Install dependencies
pip install -r training/requirements.txt

# 2. (Optional) Sign up for Mapillary Vistas Research and unpack under
#    data/raw/mapillary/{training,validation}/. Then:
export MAPILLARY_TOKEN=1

# 3. Run the full pipeline (downloads ≈ 25 GB, needs ≥ 40 GB free)
python training/data_prep/make_dataset.py

# 3a. Dry run without ClearML upload
python training/data_prep/make_dataset.py --no-upload
```

The final step prints the new ClearML `dataset_id`. Paste it into
`training/config.yaml` under `dataset_id:` to consume it in training.

## Structure

| Module | Responsibility |
|--------|---------------|
| `taxonomy.py` | 48-class index + source→target maps (single source of truth) |
| `download_coco.py` | Fetch COCO 2017, convert JSON annotations to YOLO-txt |
| `download_open_images.py` | Fetch OIv7 for door/stairs/pole, cap at 3000 instances/class |
| `download_mapillary.py` | Convert Mapillary polygons to tight bboxes |
| `merge_to_yolo.py` | Write source lineage manifest |
| `make_splits.py` | Source-stratified 90/10 split + `data.yaml` |
| `compute_class_weights.py` | Inverse-frequency class weights (mean-1 normalized) |
| `validate_dataset.py` | Hard gates (pairing, sanity, disjointness, coverage) + soft gates (balance warnings, sample overlays) |
| `upload_to_clearml.py` | Upload to ClearML with license guard |
| `make_dataset.py` | End-to-end orchestrator |

## Tests

```bash
python -m pytest training/data_prep/tests/ -v
```

All pure-logic units (bbox conversions, stratified split, class weights,
validation) have TDD coverage. Network fetchers are verified manually by
running `make_dataset.py`.

## Known limitations

- Perspective bias: OIv7 doors are mostly interior close-ups; Mapillary is
  all dashcam. Real user phone-at-chest perspective is closest to
  Mapillary. Expect mAP on `door` / `stairs` to trail the COCO classes
  until user-captured footage is added in a later dataset version.
- No test split. Val metrics are the final number.
- Mapillary license is non-commercial — do not ship paid products built on
  this dataset without re-training on a commercial substitute.
```

- [ ] **Step 3: Commit**

```bash
git add training/data_prep/README.md
git commit -m "docs(data_prep): README with quickstart + module map

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Self-review notes

**Spec coverage check** (against `docs/superpowers/specs/2026-04-18-fastlane-dataset-design.md`):

| Spec section | Implemented in |
|--------------|----------------|
| §2.1 Taxonomy (48 classes) | Task 1 (`taxonomy.py` + tests) |
| §3.1 Data flow order | Task 11 (`make_dataset.py` orchestrator) |
| §4.1 COCO conversion | Task 2 |
| §4.2 OIv7 + 3000 cap | Task 3 |
| §4.3 Mapillary polygon→bbox | Task 4 |
| §4.4 Merge + source manifest | Task 5 |
| §4.5 Stratified split seed=42 | Task 6 |
| §4.6 Class weights mean-1 | Task 7 |
| §5 Hard gates (1-4) | Task 8 |
| §5 Soft gates (5-8) | Task 9 |
| §6 Error handling (retry, token, licenses) | Tasks 2 (retry), 4 (token), 10 (license guard) |
| §6 Disk-space check | Task 11 (`_check_disk`) |
| §7 Limitations documented | Task 12 (README) |
| §8 ClearML deliverable + `dataset_id` instruction | Task 10 + 11 + 12 (README) |

All spec requirements mapped to tasks. No gaps.

**Scope check:** Training-side changes (class-weighted loss wiring) explicitly deferred to the training spec per §8 — not in this plan.

**Type consistency check:** All functions that appear in multiple modules use consistent signatures (`coco_ann_to_yolo`, `oiv7_row_to_yolo`, `polygon_annotation_to_yolo` all return `Optional[str]`; `count_instances_per_class(dir, nc)` signature matches in Task 7 test and Task 8 caller via `write_class_weights`; `ValidationError` raised in all hard gates).

---

**Plan complete and saved to `docs/superpowers/plans/2026-04-18-fastlane-dataset.md`. Two execution options:**

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, two-stage review between tasks (spec compliance, then code quality), fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints for review.

Which approach?
