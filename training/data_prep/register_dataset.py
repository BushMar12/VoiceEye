"""Register data/final/ to ClearML by reference (no file upload).

Uses Dataset.add_external_files with a file:// URI. ClearML stores only
the manifest of paths + sizes; the 35 GB of images stays on disk.
get_local_copy() resolves to the original path on the same host.
"""
from __future__ import annotations

from pathlib import Path

from clearml import Dataset


def _to_file_uri(path: Path) -> str:
    """Return RFC 8089 file:// URI for an absolute local path.
    On Windows, `C:\\foo\\bar` -> `file:///C:/foo/bar/`.
    """
    abs_path = path.resolve()
    posix = abs_path.as_posix()
    if not posix.endswith("/"):
        posix += "/"
    return f"file:///{posix.lstrip('/')}"


def register_by_reference(
    final_dir: Path,
    project: str = "VoiceEye",
    name: str = "voiceeye-yolo",
    version: str = "1.0.0",
    parent_dataset_id: str | None = None,
) -> str:
    """Register `final_dir` to ClearML as a versioned external dataset.

    Returns the new dataset_id. No file bytes are uploaded; only the
    manifest of relative paths is stored on the ClearML server.
    """
    final_dir = Path(final_dir)
    if not (final_dir / "data.yaml").exists():
        raise FileNotFoundError(
            f"data.yaml not found in {final_dir}. Refusing to register."
        )

    ds = Dataset.create(
        dataset_name=name,
        dataset_project=project,
        dataset_version=version,
        parent_datasets=[parent_dataset_id] if parent_dataset_id else None,
        description=(
            "Reference-only dataset — files live on the agent host at "
            f"{final_dir.resolve()}. Use Dataset.get(...).get_local_copy() "
            "to resolve back to the local path."
        ),
    )
    ds.add_external_files(source_url=_to_file_uri(final_dir), wildcard="*")
    ds.upload()
    ds.finalize()
    return ds.id


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--final-dir", default="data/final")
    p.add_argument("--project", default="VoiceEye")
    p.add_argument("--name", default="voiceeye-yolo")
    p.add_argument("--version", default="1.0.0")
    p.add_argument("--parent", default=None,
                   help="Parent dataset_id for lineage (optional)")
    args = p.parse_args()

    ds_id = register_by_reference(
        final_dir=Path(args.final_dir),
        project=args.project,
        name=args.name,
        version=args.version,
        parent_dataset_id=args.parent,
    )
    print(f"[clearml] Registered dataset_id = {ds_id}")
    print(f"[clearml] Set `dataset_id: \"{ds_id}\"` in training/config.yaml")
