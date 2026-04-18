"""Upload data/final/ to ClearML as voiceeye-fastlane-v1."""

from __future__ import annotations
import json
from pathlib import Path

from clearml import Dataset


REQUIRED_LICENSES = ('coco.txt', 'oiv7.txt', 'mapillary.txt')


def upload(final_dir: Path, name: str = 'voiceeye-fastlane-v1',
           project: str = 'VoiceEye') -> str:
    final_dir = Path(final_dir)

    license_dir = final_dir / 'LICENSES'
    missing = [f for f in REQUIRED_LICENSES if not (license_dir / f).exists()]
    if missing:
        raise RuntimeError(
            f"Missing license files: {missing}. Refusing to upload — "
            f"attribution is required by the source dataset terms."
        )

    ds = Dataset.create(dataset_name=name, dataset_project=project)
    ds.add_files(path=str(final_dir))

    # Add manifest with provenance
    manifest = {
        'name': name,
        'sources': ['COCO 2017', 'Open Images v7', 'Mapillary Vistas Research'],
        'nc': 48,
        'split': {'train_frac': 0.9, 'val_frac': 0.1, 'seed': 42},
        'license': 'non-commercial (Mapillary Research Edition)',
    }
    manifest_path = final_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))

    ds.upload()
    ds.finalize()
    print(f"[clearml] dataset_id = {ds.id}")
    print(f"[clearml] Set `dataset_id: \"{ds.id}\"` in training/config.yaml")
    return ds.id


if __name__ == '__main__':
    upload(Path('data/final'))
