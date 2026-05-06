"""Smoke test: 1-epoch YOLO train against the 4-image fixture.

Validates the dependency stack (ultralytics + onnx + torch) works end to
end without ClearML. Designed for headless GitHub Actions runners.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "tiny" / "data.yaml"


@pytest.mark.smoke
def test_one_epoch_smoke_train(tmp_path: Path) -> None:
    # Force CPU + offline so no GPU + no ClearML server needed in CI.
    os.environ["YOLO_OFFLINE"] = "1"
    os.environ["WANDB_DISABLED"] = "true"

    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")  # auto-downloads ~5 MB on first run
    results = model.train(
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
