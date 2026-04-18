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
