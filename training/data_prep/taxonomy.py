"""48-class taxonomy and source→target class maps for VoiceEye Fast Lane."""

# Final class order. Index = class id written to YOLO label files.
CLASSES = [
    # 42 kept from COCO (preserving a sensible order, not COCO's)
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
    'traffic light', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog',
    'backpack', 'umbrella', 'handbag', 'suitcase',
    'bottle', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'keyboard', 'cell phone',
    'microwave', 'oven', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors',
    # 6 new navigation-critical classes
    'stairs', 'curb', 'crosswalk', 'pole', 'bollard', 'door',
]
assert len(CLASSES) == 48, f"CLASSES must have 48 entries, got {len(CLASSES)}"

# Keys are COCO 2017 category_ids (gappy, max 90). Values are our CLASSES indices.
# Dropped: airplane(5), train(7), boat(9), fire hydrant(11), horse(19), sheep(20),
# cow(21), elephant(22), bear(23), zebra(24), giraffe(25), tie(32), frisbee(34),
# skis(35), snowboard(36), sports ball(37), kite(38), baseball bat(39),
# baseball glove(40), skateboard(41), surfboard(42), tennis racket(43),
# wine glass(46), banana(52), apple(53), sandwich(54), orange(55), broccoli(56),
# carrot(57), hot dog(58), pizza(59), donut(60), cake(61), remote(75),
# toaster(80), teddy bear(88), hair drier(89), toothbrush(90).
_COCO_NAME_TO_ID = {
    'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'bus': 6, 'truck': 8,
    'traffic light': 10, 'stop sign': 13, 'parking meter': 14, 'bench': 15,
    'bird': 16, 'cat': 17, 'dog': 18,
    'backpack': 27, 'umbrella': 28, 'handbag': 31, 'suitcase': 33,
    'bottle': 44, 'cup': 47, 'fork': 48, 'knife': 49, 'spoon': 50, 'bowl': 51,
    'chair': 62, 'couch': 63, 'potted plant': 64, 'bed': 65,
    'dining table': 67, 'toilet': 70,
    'tv': 72, 'laptop': 73, 'mouse': 74, 'keyboard': 76, 'cell phone': 77,
    'microwave': 78, 'oven': 79, 'sink': 81, 'refrigerator': 82,
    'book': 84, 'clock': 85, 'vase': 86, 'scissors': 87,
}
COCO_KEEP_MAP = {
    coco_id: CLASSES.index(name) for name, coco_id in _COCO_NAME_TO_ID.items()
}
assert len(COCO_KEEP_MAP) == 42

# Open Images v7 uses human-readable class names in the boxes CSV.
OIV7_CLASS_MAP = {
    'Door':   CLASSES.index('door'),
    'Stairs': CLASSES.index('stairs'),
    'Pole':   CLASSES.index('pole'),
}

# Mapillary Vistas v2.0 research class names (verify at pipeline run time —
# the Mapillary downloader prints a sample list of class strings if a key
# does not match).
MAPILLARY_CLASS_MAP = {
    'construction--flat--crosswalk-plain': CLASSES.index('crosswalk'),
    'object--support--pole':               CLASSES.index('bollard'),
    'construction--flat--curb':            CLASSES.index('curb'),
}
