# VoiceEye Fast Lane Dataset — Design Spec

**Date:** 2026-04-18
**Status:** Approved (awaiting user review before plan)
**Scope:** Retrain YOLO26n for VoiceEye's Fast Lane on a dataset curated for visually-impaired users in indoor + outdoor scenes. Public-data-only sourcing; personal / research use only.

---

## 1. Goal & context

VoiceEye's current Fast Lane runs stock YOLO26n (COCO-80). COCO is optimized for general consumer photography — it covers pets, sports, and kitchen items well, but under-indexes the things a visually-impaired user *actually* needs to know about: stairs, curbs, crosswalks, poles, bollards, and doors. This spec defines the dataset for a retraining pass that keeps the useful COCO classes and extends them with six navigation-critical classes.

**Not in this spec:** the training job itself (hyperparameters, schedule, quality gates) — that will be a separate spec that consumes this dataset.

**Target output:** a ClearML Dataset named `voiceeye-fastlane-v1` that `training/train.py` can consume by updating a single `dataset_id` in `training/config.yaml`.

---

## 2. Decisions locked

| Decision | Value | Rationale |
| :--- | :--- | :--- |
| Taxonomy | 48 classes — 42 kept from COCO + 6 new | COCO 40/80 classes are dead weight for VoiceEye (skis, toothbrush, hair drier…). 6 new classes cover navigation gaps. Signage text dropped — Slow Lane handles text via "Voice Eye, read". |
| Strategy | Extend COCO taxonomy, fine-tune from pretrained YOLO26n | From-scratch is infeasible for a personal project. Fine-tuning reuses COCO features for the 42 kept classes and learns the 6 new ones on smaller samples. |
| Sourcing | Public datasets only | Option A from brainstorm — user has no time for capture / labelling. |
| Primary sources | COCO 2017 + Open Images v7 + Mapillary Vistas Research | COCO covers the 42 kept classes; OIv7 covers `door`, `stairs`, `pole`; Mapillary covers `crosswalk`, `bollard`, `curb`. |
| License scope | Personal / research use only | Mapillary Research Edition is non-commercial. Documented in model card. |
| Imbalance correction | Class-weighted loss (inverse-frequency, mean-1 normalized) | Single config tweak; no pipeline changes. Avoids the oversampling complexity of WeightedRandomSampler. |
| Per-source caps | COCO full; OIv7 3000 instances/class; Mapillary full | Prevents OIv7 from dominating the 3 shared classes via sheer volume. |
| Splits | 90/10 train/val, source-stratified, seed=42 | Val reflects all three perspective regimes. No test set — not enough public data to hold out a third slice. |
| Deliverable | ClearML Dataset `voiceeye-fastlane-v1` | Consumed by `training/config.yaml:dataset_id`. |

### 2.1 Class taxonomy (48)

**Kept from COCO (42)** — index matches final dataset, not COCO's:

```
person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign,
parking meter, bench, bird, cat, dog, backpack, umbrella, handbag,
suitcase, bottle, cup, fork, knife, spoon, bowl, chair, couch,
potted plant, bed, dining table, toilet, tv, laptop, mouse, keyboard,
cell phone, microwave, oven, sink, refrigerator, book, clock, vase,
scissors
```

**New (6):** `stairs, curb, crosswalk, pole, bollard, door`.

**Dropped from COCO (38):** airplane, train, boat, fire hydrant, horse, sheep, cow, elephant, bear, zebra, giraffe, tie, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, wine glass, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, remote, toaster, teddy bear, hair drier, toothbrush.

Taxonomy and source→target mapping live in `training/data_prep/taxonomy.py` — single source of truth for every downstream step.

---

## 3. Architecture

**Layout:**

```
training/data_prep/
├── __init__.py
├── taxonomy.py                 # 48-class index + per-source class maps
├── download_coco.py            # COCO 2017 → filter to 42 kept classes
├── download_open_images.py     # OIv7 subset: door, stairs, pole (cap 3000/class)
├── download_mapillary.py       # Mapillary Vistas Research: crosswalk, bollard, curb
├── merge_to_yolo.py            # Stitch sources → data/processed/{images,labels}
├── make_splits.py              # Source-stratified 90/10 + data.yaml
├── compute_class_weights.py    # inverse-frequency → class_weights.yaml
├── validate_dataset.py         # QA gates (hard + soft)
├── upload_to_clearml.py        # Dataset.create() → finalize()
└── make_dataset.py             # Orchestrator (runs all steps in order)
```

**Working directories:**

```
data/
├── raw/         # Original source downloads (not uploaded)
│   ├── coco/
│   ├── oiv7/
│   └── mapillary/
├── processed/   # Merged YOLO-format dataset, pre-split
│   ├── images/
│   ├── labels/
│   └── source_manifest.csv     # image → source lineage
├── final/       # What gets uploaded to ClearML
│   ├── images/{train,val}/
│   ├── labels/{train,val}/
│   ├── data.yaml
│   ├── class_weights.yaml
│   ├── manifest.json
│   └── LICENSES/
│       ├── coco.txt
│       ├── oiv7.txt
│       └── mapillary.txt
└── qa/          # Validation outputs (not uploaded)
    ├── samples/                # 20 random bbox-overlay renders
    └── report.json             # per-class / per-source counts
```

Each stage is **idempotent** — re-running after a partial failure skips stages whose outputs already exist and match expected checksums.

### 3.1 Data flow

```
download_coco ──┐
download_oiv7 ──┼──► merge_to_yolo ──► make_splits ──► compute_class_weights ──► validate_dataset ──► upload_to_clearml
download_mapy ──┘
```

Orchestrated by `make_dataset.py`. One-shot:

```bash
python training/data_prep/make_dataset.py
```

---

## 4. Per-source conversion

### 4.1 COCO 2017

- Fetch `train2017.zip`, `val2017.zip`, `annotations_trainval2017.zip` (~19 GB total).
- Parse `instances_train2017.json` + `instances_val2017.json`.
- For each annotation: look up `category_id` in `taxonomy.coco_keep_map`; drop if not in the 42.
- Convert `bbox=[x, y, w, h]` (pixel, top-left anchor) → YOLO `cx cy w h` normalized by `image_width` / `image_height`.
- Skip annotations with `iscrowd=1` (segment masks, ambiguous bboxes).
- Images with zero kept annotations are still copied (negative samples — useful).
- Output: one `.txt` per image at `data/processed/labels/coco_<id>.txt`.

### 4.2 Open Images v7

- Use Google's `oi_download_dataset` CLI with `--classes Door Stairs Pole --type_csv detection`.
- OIv7 boxes are already normalized [0,1] as `XMin, XMax, YMin, YMax` — convert to `cx cy w h`.
- **Cap per class at 3000 instances.** After download, sort images by `IsGroupOf=0` first, then by bbox area descending; accept images into the pool until the per-class quota is hit; discard the rest. Images are kept/dropped as a unit (never partially).
- `IsGroupOf=1` boxes (crowds) are skipped — loose / inaccurate.
- Output: `data/processed/labels/oiv7_<image_id>.txt`.

### 4.3 Mapillary Vistas Research

- Requires `MAPILLARY_TOKEN` env var (Research Edition signup gate).
- Polygon-segmentation labels. For each polygon belonging to `crosswalk-plain`, `bollard`, `curb` (the 3 we want), compute tight axis-aligned bbox: `min(x), min(y), max(x), max(y)` over polygon vertices.
- Drop bboxes with normalized area < 0.001 (noise / occluded fragments).
- Convert to YOLO `cx cy w h` normalized form.
- Other Mapillary classes (including `pole-utility` — already covered by OIv7) are skipped.
- Output: `data/processed/labels/mapillary_<image_id>.txt`.

### 4.4 Merge

`merge_to_yolo.py` concatenates the three source directories into `data/processed/{images,labels}/` and writes `source_manifest.csv`:

```
filename,source,original_id
coco_000000139.jpg,coco,139
oiv7_abc123.jpg,oiv7,abc123
mapillary_xyz789.jpg,mapillary,xyz789
```

### 4.5 Source-stratified split

Naïve random 90/10 over the merged pool would still give each source ≈90/10, but variance on the small Mapillary slice is high enough that we pin the RNG seed per source:

```python
for source in ['coco', 'oiv7', 'mapillary']:
    images = [img for img in pool if img.source == source]
    random.Random(42).shuffle(images)
    cut = int(len(images) * 0.9)
    train += images[:cut]
    val   += images[cut:]
```

Seed 42 is fixed — re-runs produce byte-identical splits, required for deterministic ClearML versioning.

`make_splits.py` writes `data/final/data.yaml`:

```yaml
path: .
train: images/train
val:   images/val
nc: 48
names: [person, bicycle, car, ...]   # from taxonomy.py
```

### 4.6 Class weights

Inverse-frequency, smoothed, mean-1 normalized:

```python
counts = count_instances_per_class(labels_train)          # dict[int, int]
inv    = {c: 1.0 / max(n, 1) for c, n in counts.items()}
total  = sum(inv.values())
weights = {c: (inv[c] / total) * nc for c in range(nc)}   # mean weight = 1.0
```

Written to `data/final/class_weights.yaml` with both `id` and `name` alongside the weight for human review. Mean-1 normalization keeps overall loss magnitude unchanged so existing LR schedules don't need re-tuning.

---

## 5. QA gates (`validate_dataset.py`)

### Hard gates (fail build)

1. **Image-label pairing:** every `labels/**/*.txt` has a matching `images/**/*.{jpg,png}` and vice versa.
2. **Bbox sanity:** every label line parses as `cls cx cy w h`, `cls ∈ [0, 48)`, all coords in `[0, 1]`, `w > 0 ∧ h > 0`.
3. **Split disjointness:** no image filename appears in both train and val.
4. **Class coverage:** every one of the 48 classes has ≥ 1 instance in train. Extinct class would divide by zero in weight calc — loud failure is correct.

### Soft gates (warn, don't fail)

5. **Class balance report:** print instances/class sorted ascending. Warn if any train class < 100 instances.
6. **Val class coverage:** warn if any val class < 10 instances (mAP noisy for it).
7. **Sample overlays:** render 20 random images with bboxes drawn to `data/qa/samples/` for eyeball check.
8. **Report:** write `data/qa/report.json` with counts per source × per split × per class.

---

## 6. Error handling

- **Network failures** (any source download): retry 3× with exponential backoff (1 s / 4 s / 16 s), then fail with URL + HTTP code. Truncated archives are deleted — no partial-file acceptance.
- **Missing `MAPILLARY_TOKEN`:** print the Research Edition signup URL and skip the Mapillary step. `validate_dataset.py`'s hard gate on class coverage will then fail — exactly the loud failure we want.
- **License files:** each source's license text is copied verbatim to `data/final/LICENSES/{coco,oiv7,mapillary}.txt`. `upload_to_clearml.py` refuses to upload if any of the three is missing — prevents accidentally publishing without attribution.
- **Disk pressure:** total raw downloads ≈ 25 GB. The orchestrator exits early with a clear message if `shutil.disk_usage()` reports < 40 GB free before starting.

---

## 7. Known limitations

- **Perspective bias.** OIv7 doors are mostly interior close-ups at ~1.6 m eye height. Mapillary is all dashcam at ~1.5 m facing forward. Real users hold the phone at ~1.2 m facing forward — closer to dashcam than to OIv7. Expect mAP on `door` and `stairs` to be weaker than the 42 COCO classes until user-captured footage extends the training pool in a future dataset version.
- **No test set.** 90/10 split only. Final mAP is reported on val. A true test set requires held-out footage that doesn't exist yet in public sources.
- **Class-weighted loss ≠ more data.** Weights reduce bias but not variance — under-represented classes (`stairs`, `bollard`) will still be harder to detect even after weighting. The model card must flag this.
- **Mapillary license.** Research Edition is non-commercial. Any ONNX model trained on this dataset is therefore non-commercial. Fine for the user's personal / research use; must not ship to paying users without re-training on a commercial-licensed substitute.

---

## 8. Deliverable

ClearML Dataset `voiceeye-fastlane-v1` containing:

```
images/{train,val}/            # The actual frames
labels/{train,val}/            # YOLO-txt labels
data.yaml                      # Ultralytics dataset manifest
class_weights.yaml             # 48 weights, mean = 1.0
manifest.json                  # Full provenance: sources, counts, checksums, seed
LICENSES/{coco,oiv7,mapillary}.txt
```

Consumed by `training/config.yaml` by setting `dataset_id` to the new UUID returned from `Dataset.finalize()`. No other training-side changes in this spec — class-weighted loss wiring is deferred to the training spec.
