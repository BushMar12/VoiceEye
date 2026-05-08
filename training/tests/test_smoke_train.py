"""Smoke test: 1-epoch YOLO train against the 4-image fixture.

Validates the dependency stack (ultralytics + onnx + torch) works end to
end without ClearML. Designed for headless GitHub Actions runners.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tiny"


@pytest.mark.smoke
def test_one_epoch_smoke_train(tmp_path: Path) -> None:
    # Force CPU + offline so no GPU + no ClearML server needed in CI.
    os.environ["YOLO_OFFLINE"] = "1"
    os.environ["WANDB_DISABLED"] = "true"

    from ultralytics import YOLO

    # Ultralytics resolves the `path:` field of data.yaml relative to its
    # configured datasets dir (defaults to `<cwd>/datasets`), not relative
    # to the yaml file. The committed fixture has `path: .` which works
    # only when cwd happens to equal the fixture dir. Rewrite to an
    # absolute path so the test works regardless of where pytest was
    # invoked from (relevant for CI runners).
    rewritten_yaml = tmp_path / "data.yaml"
    rewritten_yaml.write_text(
        f"path: {FIXTURE_DIR.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names:\n"
        "  - obj\n"
    )

    model = YOLO("yolo26n.pt")  # auto-downloads ~5 MB on first run
    results = model.train(
        data=str(rewritten_yaml),
        epochs=1,
        imgsz=64,
        batch=2,
        device="cpu",
        workers=0,
        project=str(tmp_path),
        name="smoke",
        exist_ok=True,
        plots=False,
        verbose=False,
    )

    # The trainer must produce a finite training loss.
    assert results is not None
    final_loss = float(model.trainer.loss)
    assert math.isfinite(final_loss), f"non-finite loss: {final_loss}"

    # And a best.pt checkpoint must be on disk.
    best = Path(model.trainer.best)
    assert best.exists(), f"best.pt not produced at {best}"
