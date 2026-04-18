import pytest
from training.data_prep.download_mapillary import (
    polygon_to_bbox,
    polygon_annotation_to_yolo,
)


def test_polygon_to_bbox_basic():
    # Square polygon at pixel coords
    poly = [(10, 20), (30, 20), (30, 40), (10, 40)]
    x0, y0, x1, y1 = polygon_to_bbox(poly)
    assert (x0, y0, x1, y1) == (10, 20, 30, 40)


def test_polygon_to_bbox_irregular():
    poly = [(5, 5), (15, 10), (12, 25), (3, 20)]
    x0, y0, x1, y1 = polygon_to_bbox(poly)
    assert (x0, y0, x1, y1) == (3, 5, 15, 25)


def test_polygon_annotation_mapped_class():
    ann = {
        'label': 'construction--flat--crosswalk-plain',
        'polygon': [(100, 200), (300, 200), (300, 280), (100, 280)],
    }
    line = polygon_annotation_to_yolo(ann, img_w=1000, img_h=800)
    parts = line.split()
    # 'crosswalk' is class 44 in CLASSES
    assert parts[0] == '44'
    # bbox: (100,200,300,280) → cx=0.2, cy=0.3, w=0.2, h=0.1
    assert float(parts[1]) == pytest.approx(0.2)
    assert float(parts[2]) == pytest.approx(0.3)
    assert float(parts[3]) == pytest.approx(0.2)
    assert float(parts[4]) == pytest.approx(0.1)


def test_polygon_annotation_unknown_class_returns_none():
    ann = {'label': 'sky', 'polygon': [(0, 0), (10, 0), (10, 10), (0, 10)]}
    assert polygon_annotation_to_yolo(ann, 100, 100) is None


def test_polygon_annotation_tiny_bbox_returns_none():
    # Area = 1/10000 = 0.0001 < 0.001 threshold
    ann = {
        'label': 'construction--flat--crosswalk-plain',
        'polygon': [(0, 0), (1, 0), (1, 1), (0, 1)],
    }
    assert polygon_annotation_to_yolo(ann, 100, 100) is None
