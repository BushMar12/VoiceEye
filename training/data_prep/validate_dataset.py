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
