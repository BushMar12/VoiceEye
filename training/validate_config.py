"""
Validate training/config.yaml for correctness.

Used by GitHub Actions CI to prevent regressions:
- Wrong model name (e.g. yolov8n instead of yolo26n)
- Wrong export format (e.g. tfjs instead of onnx)
- Invalid numeric ranges

Usage:
    python training/validate_config.py
    python training/validate_config.py --config path/to/config.yaml
"""

import argparse
import sys

import yaml

REQUIRED_FIELDS = [
    "clearml_project",
    "clearml_task_name",
    "dataset_id",
    "model_weights",
    "epochs",
    "batch",
    "imgsz",
    "optimizer",
    "lr0",
    "export_format",
    "export_imgsz",
    "half",
    "min_map50",
]

ERRORS: list[str] = []


def error(msg: str) -> None:
    ERRORS.append(msg)
    print(f"  FAIL: {msg}")


def check(condition: bool, msg: str) -> None:
    if not condition:
        error(msg)
    else:
        print(f"  OK:   {msg}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config.yaml")
    args = parser.parse_args()

    print(f"Validating {args.config}...\n")

    try:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: {args.config} not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"ERROR: Invalid YAML: {e}")
        sys.exit(1)

    # Required fields
    print("[Required fields]")
    for field in REQUIRED_FIELDS:
        check(field in cfg, f"'{field}' exists")

    # Model name — must be YOLO26n, not yolov8n
    print("\n[Model]")
    weights = cfg.get("model_weights", "")
    check("26n" in weights.lower(),
          f"model_weights contains '26n' (got: '{weights}')")

    # Export format — must be ONNX, not TFJS
    print("\n[Export]")
    fmt = cfg.get("export_format", "")
    check(fmt == "onnx",
          f"export_format is 'onnx' (got: '{fmt}')")
    check(cfg.get("half") is False,
          f"half is false (WASM requires float32)")

    # Numeric ranges
    print("\n[Numeric ranges]")
    epochs = cfg.get("epochs", 0)
    check(isinstance(epochs, int) and epochs > 0,
          f"epochs > 0 (got: {epochs})")
    check(isinstance(epochs, int) and epochs >= 50,
          f"epochs >= 50 for convergence (got: {epochs})")

    imgsz = cfg.get("imgsz", 0)
    check(320 <= imgsz <= 1280,
          f"imgsz in [320, 1280] (got: {imgsz})")

    batch = cfg.get("batch", 0)
    check(batch > 0,
          f"batch > 0 (got: {batch})")

    lr0 = cfg.get("lr0", 0)
    check(0 < lr0 <= 0.1,
          f"lr0 in (0, 0.1] (got: {lr0})")

    min_map = cfg.get("min_map50", 0)
    check(0 <= min_map <= 1,
          f"min_map50 in [0, 1] (got: {min_map})")

    patience = cfg.get("patience", 0)
    check(patience > 0,
          f"patience > 0 (got: {patience})")

    # Summary
    print(f"\n{'='*50}")
    if ERRORS:
        print(f"VALIDATION FAILED — {len(ERRORS)} error(s)")
        sys.exit(1)
    else:
        print("VALIDATION PASSED — all checks OK")


if __name__ == "__main__":
    main()
