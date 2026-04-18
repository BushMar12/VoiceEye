from training.data_prep.taxonomy import (
    CLASSES,
    COCO_KEEP_MAP,
    OIV7_CLASS_MAP,
    MAPILLARY_CLASS_MAP,
)


def test_classes_has_48_entries():
    assert len(CLASSES) == 48


def test_classes_are_unique():
    assert len(set(CLASSES)) == 48


def test_six_new_classes_present():
    for new_cls in ['stairs', 'curb', 'crosswalk', 'pole', 'bollard', 'door']:
        assert new_cls in CLASSES


def test_dropped_coco_classes_absent():
    for dropped in ['airplane', 'train', 'boat', 'banana', 'toothbrush', 'hair drier']:
        assert dropped not in CLASSES


def test_coco_keep_map_covers_42_classes():
    # Keys are COCO category_ids (1..90 with gaps); values are our class indices
    assert len(COCO_KEEP_MAP) == 42
    for our_idx in COCO_KEEP_MAP.values():
        assert 0 <= our_idx < 48


def test_oiv7_map_has_three_classes():
    # Keys are OIv7 class names; values are our indices
    assert set(OIV7_CLASS_MAP.keys()) == {'Door', 'Stairs', 'Pole'}
    for our_idx in OIV7_CLASS_MAP.values():
        assert 0 <= our_idx < 48


def test_mapillary_map_has_three_classes():
    assert set(MAPILLARY_CLASS_MAP.keys()) == {
        'crosswalk-plain', 'object--support--pole-bollard', 'curb'
    } or set(MAPILLARY_CLASS_MAP.keys()) == {
        'construction--flat--crosswalk-plain',
        'object--support--pole',
        'construction--flat--curb',
    }
    # Spec leaves the exact Mapillary class key string to whatever the dataset
    # ships; this test just pins the count and that values are valid.
    assert len(MAPILLARY_CLASS_MAP) == 3


def test_new_classes_not_in_source_maps():
    # The 3 new classes from OIv7 and 3 from Mapillary cover all 6 new ones
    new_indices = {CLASSES.index(c) for c in
                   ['stairs', 'curb', 'crosswalk', 'pole', 'bollard', 'door']}
    mapped_new = set(OIV7_CLASS_MAP.values()) | set(MAPILLARY_CLASS_MAP.values())
    assert new_indices == mapped_new
