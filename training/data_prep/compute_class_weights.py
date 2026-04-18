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
