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
