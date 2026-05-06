"""Tests for register_dataset.register_by_reference."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_register_by_reference_calls_add_external_files(tmp_path: Path) -> None:
    """register_by_reference must add external files (no add_files) and
    finalize a v1.0.0 dataset under the expected project."""
    from training.data_prep.register_dataset import register_by_reference

    # Minimal fake dataset on disk (file presence is all the function checks)
    (tmp_path / "data.yaml").write_text("path: .\nnc: 1\nnames: [obj]\n")
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "images" / "train" / "a.jpg").write_bytes(b"\x00")
    (tmp_path / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 1 1\n")

    fake_ds = MagicMock()
    fake_ds.id = "deadbeef"
    with patch("training.data_prep.register_dataset.Dataset") as mock_dataset:
        mock_dataset.create.return_value = fake_ds
        result_id = register_by_reference(
            final_dir=tmp_path,
            project="VoiceEye",
            name="voiceeye-yolo",
        )

    assert result_id == "deadbeef"
    mock_dataset.create.assert_called_once()
    fake_ds.add_external_files.assert_called_once()
    # External URI must be a file:// URI rooted at final_dir
    args, kwargs = fake_ds.add_external_files.call_args
    src = kwargs.get("source_url") or args[0]
    assert src.startswith("file:///"), f"expected file:// URI, got {src}"
    fake_ds.upload.assert_called_once()
    fake_ds.finalize.assert_called_once()
    # Must NOT call add_files (which would copy bytes)
    fake_ds.add_files.assert_not_called()


def test_register_by_reference_requires_data_yaml(tmp_path: Path) -> None:
    """If data.yaml is missing, refuse to register."""
    from training.data_prep.register_dataset import register_by_reference

    with pytest.raises(FileNotFoundError, match="data.yaml"):
        register_by_reference(final_dir=tmp_path)
