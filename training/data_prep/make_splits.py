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
