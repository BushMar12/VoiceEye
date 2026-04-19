# Fast Lane Dataset Builder

Builds `voiceeye-fastlane-v1` — a 48-class YOLO dataset for VoiceEye's
Fast Lane, merged from three public sources.

See the [design spec](../../docs/superpowers/specs/2026-04-18-fastlane-dataset-design.md)
for background and rationale.

## Quickstart

```bash
# 1. Install dependencies
pip install -r training/requirements.txt

# 2. (Optional) Sign up for Mapillary Vistas Research and unpack under
#    data/raw/mapillary/{training,validation}/. Then:
export MAPILLARY_TOKEN=1

# 3. Run the full pipeline (downloads ≈ 25 GB, needs ≥ 40 GB free)
python training/data_prep/make_dataset.py

# 3a. Dry run without ClearML upload
python training/data_prep/make_dataset.py --no-upload
```

The final step prints the new ClearML `dataset_id`. Paste it into
`training/config.yaml` under `dataset_id:` to consume it in training.

## Structure

| Module | Responsibility |
|--------|---------------|
| `taxonomy.py` | 48-class index + source→target maps (single source of truth) |
| `download_coco.py` | Fetch COCO 2017, convert JSON annotations to YOLO-txt |
| `download_open_images.py` | Fetch OIv7 for door/stairs/pole, cap at 3000 instances/class |
| `download_mapillary.py` | Convert Mapillary polygons to tight bboxes |
| `merge_to_yolo.py` | Write source lineage manifest |
| `make_splits.py` | Source-stratified 90/10 split + `data.yaml` |
| `compute_class_weights.py` | Inverse-frequency class weights (mean-1 normalized) |
| `validate_dataset.py` | Hard gates (pairing, sanity, disjointness, coverage) + soft gates (balance warnings, sample overlays) |
| `upload_to_clearml.py` | Upload to ClearML with license guard |
| `make_dataset.py` | End-to-end orchestrator |

## Tests

```bash
python -m pytest training/data_prep/tests/ -v
```

All pure-logic units (bbox conversions, stratified split, class weights,
validation) have TDD coverage. Network fetchers are verified manually by
running `make_dataset.py`.

## Known limitations

- Perspective bias: OIv7 doors are mostly interior close-ups; Mapillary is
  all dashcam. Real user phone-at-chest perspective is closest to
  Mapillary. Expect mAP on `door` / `stairs` to trail the COCO classes
  until user-captured footage is added in a later dataset version.
- No test split. Val metrics are the final number.
- Mapillary license is non-commercial — do not ship paid products built on
  this dataset without re-training on a commercial substitute.
