"""Run the full dataset pipeline end-to-end.

    python training/data_prep/make_dataset.py
"""

from __future__ import annotations
import shutil
from pathlib import Path

from training.data_prep.download_coco import download_and_convert_coco
from training.data_prep.download_open_images import download_and_convert_oiv7
from training.data_prep.download_mapillary import download_and_convert_mapillary
from training.data_prep.merge_to_yolo import write_source_manifest
from training.data_prep.make_splits import materialize_splits
from training.data_prep.compute_class_weights import write_class_weights
from training.data_prep.validate_dataset import validate_all
from training.data_prep.upload_to_clearml import upload


MIN_FREE_GB = 40


def _check_disk(path: Path) -> None:
    free_gb = shutil.disk_usage(str(path.resolve().anchor)).free / (1024 ** 3)
    if free_gb < MIN_FREE_GB:
        raise RuntimeError(
            f"Only {free_gb:.1f} GB free — need >= {MIN_FREE_GB} GB for raw downloads."
        )


def _copy_licenses(final_dir: Path) -> None:
    licenses_src = Path('training/data_prep/licenses')
    licenses_dst = final_dir / 'LICENSES'
    licenses_dst.mkdir(parents=True, exist_ok=True)
    for name in ('coco.txt', 'oiv7.txt', 'mapillary.txt'):
        src = licenses_src / name
        dst = licenses_dst / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def main(upload_to_clearml: bool = True) -> None:
    data = Path('data')
    _check_disk(data)

    raw = data / 'raw'
    processed = data / 'processed'
    final = data / 'final'

    # 1-3: Downloads + conversions
    download_and_convert_coco(raw / 'coco', processed / 'labels', processed / 'images')
    download_and_convert_oiv7(raw / 'oiv7', processed / 'labels', processed / 'images')
    download_and_convert_mapillary(raw / 'mapillary', processed / 'labels', processed / 'images')

    # 4: Manifest
    manifest_path = processed / 'source_manifest.csv'
    write_source_manifest(processed / 'images', manifest_path)

    # 5: Splits + data.yaml
    n_train, n_val = materialize_splits(manifest_path, processed, final)
    print(f"[split] train={n_train} val={n_val}")

    # 6: Class weights
    write_class_weights(final / 'labels' / 'train', final / 'class_weights.yaml')

    # 7: Licenses
    _copy_licenses(final)

    # 8: Validate (raises on hard gate failure)
    report = validate_all(final)
    print(f"[validate] train per-source counts: {report['train']['per_source']}")
    print(f"[validate] val   per-source counts: {report['val']['per_source']}")

    # 9: Upload
    if upload_to_clearml:
        upload(final)


if __name__ == '__main__':
    import sys
    skip_upload = '--no-upload' in sys.argv
    main(upload_to_clearml=not skip_upload)
