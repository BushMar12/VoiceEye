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
