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
