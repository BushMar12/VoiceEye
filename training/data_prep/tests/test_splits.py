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
