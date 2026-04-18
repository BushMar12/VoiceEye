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
