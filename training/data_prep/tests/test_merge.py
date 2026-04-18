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
