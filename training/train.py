"""
VoiceEye FastLane — ClearML-tracked YOLO26n training script.

All hyperparameters are loaded from config.yaml and registered with
ClearML via task.connect(), making them editable from the Web UI and
overridable by the HPO optimizer.

Usage:
    python training/train.py                          # Use config.yaml defaults
    python training/train.py --epochs 200             # CLI override
    python training/train.py --config path/to.yaml    # Custom config file
"""

import argparse
import hashlib
import sys
from pathlib import Path

import yaml
from clearml import Dataset, Task
from ultralytics import YOLO


# ── Helpers ──────────────────────────────────────────────────────────

def load_config(config_path: str = "training/config.yaml") -> dict:
    """Load config YAML and return a flat dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def coerce_types(cfg: dict, reference: dict) -> dict:
    """
    ClearML serialises connected dict values as strings when HPO
    overrides them. Coerce back to the original types using a
    reference config as the type template.
    """
    coerced = {}
    for key, value in cfg.items():
        if key in reference and isinstance(value, str):
            ref_type = type(reference[key])
            try:
                if ref_type is bool:
                    value = value.lower() in ("true", "1", "yes")
                else:
                    value = ref_type(value)
            except (ValueError, TypeError):
                pass
        coerced[key] = value
    return coerced


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train YOLO26n for VoiceEye")
    parser.add_argument("--config", default="training/config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs")
    parser.add_argument("--batch", type=int, default=None,
                        help="Override batch size")
    cli = parser.parse_args()

    # Load config with CLI overrides
    cfg = load_config(cli.config)
    reference_cfg = dict(cfg)  # snapshot for type coercion later
    if cli.epochs is not None:
        cfg["epochs"] = cli.epochs
    if cli.batch is not None:
        cfg["batch"] = cli.batch

    # ── ClearML Task ─────────────────────────────────────────────────
    task = Task.init(
        project_name=cfg["clearml_project"],
        task_name=cfg["clearml_task_name"],
    )
    # Register ALL config keys as hyperparameters.
    # HPO clones this task and overrides values here before training.
    task.connect(cfg, name="training_config")

    # After connect(), ClearML may have overridden values (e.g. from HPO).
    # Coerce string values back to their original types.
    cfg = coerce_types(cfg, reference_cfg)

    # ── Dataset ──────────────────────────────────────────────────────
    print(f"Fetching dataset {cfg['dataset_id']} from ClearML...")
    dataset = Dataset.get(dataset_id=cfg["dataset_id"])
    dataset_path = dataset.get_local_copy()
    print(f"Dataset ready at: {dataset_path}")

    data_yaml = Path(dataset_path) / "data.yaml"
    if not data_yaml.exists():
        print(f"ERROR: {data_yaml} not found in dataset.")
        task.close()
        sys.exit(1)

    # ── Train ────────────────────────────────────────────────────────
    print(f"\nTraining {cfg['model_weights']} for {cfg['epochs']} epochs "
          f"(patience={cfg['patience']}, optimizer={cfg['optimizer']})...")

    model = YOLO(cfg["model_weights"])
    results = model.train(
        data=str(data_yaml),
        epochs=cfg["epochs"],
        patience=cfg["patience"],
        batch=cfg["batch"],
        imgsz=cfg["imgsz"],
        optimizer=cfg["optimizer"],
        lr0=cfg["lr0"],
        lrf=cfg["lrf"],
        cos_lr=cfg["cos_lr"],
        warmup_epochs=cfg["warmup_epochs"],
        warmup_momentum=cfg["warmup_momentum"],
        weight_decay=cfg["weight_decay"],
        dropout=cfg["dropout"],
        hsv_h=cfg["hsv_h"],
        hsv_s=cfg["hsv_s"],
        hsv_v=cfg["hsv_v"],
        degrees=cfg["degrees"],
        translate=cfg["translate"],
        scale=cfg["scale"],
        fliplr=cfg["fliplr"],
        mosaic=cfg["mosaic"],
        mixup=cfg["mixup"],
        project="VoiceEye_Runs",
        name="fastlane_train",
        exist_ok=True,
    )
    # Ultralytics auto-logs loss curves, mAP, sample images to the
    # active ClearML task — no explicit callback code needed.

    # ── Evaluate ─────────────────────────────────────────────────────
    print("\nRunning validation...")
    metrics = model.val(
        data=str(data_yaml),
        conf=cfg["conf_threshold"],
        iou=cfg["iou_threshold"],
    )
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)

    print(f"\n{'='*50}")
    print(f"  mAP@50:    {map50:.4f}")
    print(f"  mAP@50-95: {map50_95:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"{'='*50}")

    # Report scalar for HPO objective monitoring
    task.get_logger().report_scalar(
        title="val", series="mAP50",
        value=map50, iteration=cfg["epochs"]
    )

    # ── Quality Gate ─────────────────────────────────────────────────
    min_map = cfg.get("min_map50", 0.40)
    if map50 < min_map:
        msg = (f"QUALITY GATE FAILED: mAP@50 = {map50:.4f} < "
               f"threshold {min_map:.2f}. Model rejected.")
        print(f"\n{msg}")
        task.get_logger().report_text(msg)
        task.close()
        sys.exit(1)

    print(f"\nQuality gate PASSED (mAP@50 = {map50:.4f} >= {min_map:.2f})")

    # ── Export ONNX ──────────────────────────────────────────────────
    print(f"\nExporting to {cfg['export_format'].upper()}...")
    export_path = model.export(
        format=cfg["export_format"],
        imgsz=cfg["export_imgsz"],
        half=cfg["half"],
        simplify=cfg.get("export_simplify", True),
        opset=cfg.get("export_opset", 17),
    )
    onnx_path = Path(export_path)
    checksum = sha256_file(onnx_path)
    size_mb = onnx_path.stat().st_size / (1024 * 1024)

    print(f"  Path:     {onnx_path}")
    print(f"  Size:     {size_mb:.2f} MB")
    print(f"  SHA-256:  {checksum}")

    # ── Upload Artifact ──────────────────────────────────────────────
    task.upload_artifact(
        name="fastlane_onnx_model",
        artifact_object=str(onnx_path),
        metadata={
            "map50": map50,
            "map50_95": map50_95,
            "precision": precision,
            "recall": recall,
            "dataset_id": cfg["dataset_id"],
            "epochs": cfg["epochs"],
            "optimizer": cfg["optimizer"],
            "lr0": cfg["lr0"],
            "model_size_mb": round(size_mb, 2),
            "sha256": checksum,
        },
    )
    print("\nONNX model uploaded to ClearML artifacts.")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("  TRAINING COMPLETE")
    print(f"  Model:    {cfg['model_weights']}")
    print(f"  mAP@50:   {map50:.4f}")
    print(f"  Export:   {onnx_path.name} ({size_mb:.2f} MB)")
    print(f"  Checksum: {checksum[:16]}...")
    print(f"{'='*50}")

    task.close()
    print("ClearML task closed. Done.")


if __name__ == "__main__":
    main()
