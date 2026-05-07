"""
VoiceEye FastLane — ClearML Pipeline Controller.

Chains 4 steps: data_prep -> train -> evaluate -> export.
Supports local execution (default) and remote execution via clearml-agent.

Usage:
    python training/pipeline.py                  # Local — all steps in-process
    python training/pipeline.py --remote         # Remote — steps enqueued to agent
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import yaml
from clearml import Dataset, Task
from clearml.automation.controller import PipelineController


# ═══════════════════════════════════════════════════════════════════════
# Pipeline Step Functions
# Each function is serialised by PipelineController when running remote.
# Imports must be inside the function body for remote execution.
# ═══════════════════════════════════════════════════════════════════════

def data_prep_step(dataset_id: str) -> str:
    """Fetch dataset from ClearML and validate structure."""
    from pathlib import Path

    import yaml
    from clearml import Dataset

    dataset = Dataset.get(dataset_id=dataset_id)
    dataset_path = dataset.get_local_copy()

    data_yaml = Path(dataset_path) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")

    with open(data_yaml) as f:
        data_info = yaml.safe_load(f)

    names = data_info.get("names", [])
    if isinstance(names, dict):
        names = list(names.values())

    print(f"Dataset at: {dataset_path}")
    print(f"Classes ({len(names)}): {names}")

    return dataset_path


def hpo_step(dataset_path: str, config: dict) -> dict:
    """Run a short HPO sweep against the registered template task; return
    the winning hyperparameter dict to be merged on top of `config` in
    the train step. Trial-only knobs (epochs/patience) are excluded so
    the full train uses production-length values."""
    from clearml.automation import (
        DiscreteParameterRange,
        HyperParameterOptimizer,
        UniformIntegerParameterRange,
        UniformParameterRange,
    )
    from clearml.automation.optuna import OptimizerOptuna

    template_id = config.get("hpo_template_task_id", "")
    if not template_id:
        raise ValueError(
            "hpo_template_task_id is empty in config.yaml. "
            "Create a template with `python training/train.py --epochs 5 "
            "--data data/final/data.yaml`, then paste its ClearML task ID "
            "into config.yaml before running pipeline_mode=hpo+train."
        )

    hyper_parameters = [
        UniformParameterRange("training_config/lr0",
                              min_value=0.0001, max_value=0.01, step_size=0.0001),
        UniformParameterRange("training_config/lrf",
                              min_value=0.001, max_value=0.1, step_size=0.001),
        DiscreteParameterRange("training_config/optimizer",
                               values=["SGD", "AdamW", "Adam"]),
        DiscreteParameterRange("training_config/batch",
                               values=["8", "16", "32"]),
        UniformParameterRange("training_config/weight_decay",
                              min_value=0.0001, max_value=0.001, step_size=0.0001),
        UniformIntegerParameterRange("training_config/warmup_epochs",
                                     min_value=2, max_value=10, step_size=1),
        UniformParameterRange("training_config/mosaic",
                              min_value=0.0, max_value=1.0, step_size=0.1),
        UniformParameterRange("training_config/scale",
                              min_value=0.0, max_value=0.9, step_size=0.1),
        UniformParameterRange("training_config/mixup",
                              min_value=0.0, max_value=0.3, step_size=0.05),
        DiscreteParameterRange("training_config/cos_lr",
                               values=["True", "False"]),
        DiscreteParameterRange("training_config/epochs",
                               values=[str(config.get("hpo_trial_epochs", 50))]),
        DiscreteParameterRange("training_config/patience",
                               values=[str(config.get("hpo_trial_patience", 15))]),
    ]

    optimizer = HyperParameterOptimizer(
        base_task_id=template_id,
        hyper_parameters=hyper_parameters,
        objective_metric_title="val",
        objective_metric_series="mAP50",
        objective_metric_sign="max",
        optimizer_class=OptimizerOptuna,
        max_number_of_concurrent_tasks=int(config.get("hpo_concurrent_tasks", 2)),
        total_max_jobs=int(config.get("hpo_max_trials", 20)),
        execution_queue=config.get("queue_gpu", "gpu"),
    )
    optimizer.set_report_period(2)
    optimizer.start()
    optimizer.wait()

    top = optimizer.get_top_experiments(top_k=1)
    if not top:
        raise RuntimeError(
            "HPO produced no completed trials. "
            "Check the agent on the gpu queue + the template task is valid."
        )
    best_task = top[0]
    raw = best_task.get_parameters() or {}

    out: dict = {}
    for key, value in raw.items():
        if not key.startswith("training_config/"):
            continue
        param_name = key.split("/", 1)[1]
        # Drop trial-only overrides so the full train uses config.yaml's
        # production-length epochs/patience.
        if param_name in ("epochs", "patience"):
            continue
        # Coerce to the type the original config used.
        if param_name in config:
            t = type(config[param_name])
            try:
                if t is bool:
                    value = str(value).lower() in ("true", "1", "yes")
                else:
                    value = t(value)
            except (ValueError, TypeError):
                pass
        out[param_name] = value

    print(f"[hpo] Best trial: {best_task.id}  mAP@50 maximised")
    print(f"[hpo] Winning params (overlay onto config): {out}")
    return out


def train_step(dataset_path: str, config: dict, hpo_params: dict | None = None) -> tuple:
    """Run YOLO training, return best weights path and mAP@50.
    If `hpo_params` is provided (from a preceding hpo_step), those values
    override the matching keys in `config` for this run only."""
    from pathlib import Path

    from ultralytics import YOLO

    if hpo_params:
        config = {**config, **hpo_params}
        print(f"[train] Applied HPO winners on top of config: "
              f"{sorted(hpo_params.keys())}")

    data_yaml = str(Path(dataset_path) / "data.yaml")
    model = YOLO(config["model_weights"])

    results = model.train(
        data=data_yaml,
        epochs=config["epochs"],
        patience=config["patience"],
        batch=config["batch"],
        imgsz=config["imgsz"],
        optimizer=config["optimizer"],
        lr0=config["lr0"],
        lrf=config["lrf"],
        cos_lr=config["cos_lr"],
        warmup_epochs=config["warmup_epochs"],
        warmup_momentum=config["warmup_momentum"],
        weight_decay=config["weight_decay"],
        dropout=config["dropout"],
        hsv_h=config["hsv_h"],
        hsv_s=config["hsv_s"],
        hsv_v=config["hsv_v"],
        degrees=config["degrees"],
        translate=config["translate"],
        scale=config["scale"],
        fliplr=config["fliplr"],
        mosaic=config["mosaic"],
        mixup=config["mixup"],
        project="VoiceEye_Runs",
        name="pipeline_train",
        exist_ok=True,
    )

    best_pt = str(model.trainer.best)
    map50 = results.results_dict.get("metrics/mAP50(B)", 0.0)
    print(f"Training complete. Best weights: {best_pt}, mAP@50: {map50:.4f}")

    return best_pt, float(map50)


def evaluate_step(model_path: str, dataset_path: str,
                  conf_threshold: float, iou_threshold: float,
                  min_map50: float) -> dict:
    """Validate model, check quality gate, return metrics."""
    import sys
    from pathlib import Path

    from clearml import Task
    from ultralytics import YOLO

    data_yaml = str(Path(dataset_path) / "data.yaml")
    model = YOLO(model_path)

    metrics = model.val(data=data_yaml, conf=conf_threshold, iou=iou_threshold)

    eval_results = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

    print(f"\nEvaluation Results:")
    for k, v in eval_results.items():
        print(f"  {k}: {v:.4f}")

    # Report to ClearML
    task = Task.current_task()
    if task:
        task.get_logger().report_scalar(
            "val", "mAP50", value=eval_results["mAP50"], iteration=1
        )

    # Quality gate
    if eval_results["mAP50"] < min_map50:
        msg = (f"QUALITY GATE FAILED: mAP@50 = {eval_results['mAP50']:.4f} "
               f"< threshold {min_map50:.2f}")
        print(f"\n{msg}")
        raise ValueError(msg)

    print(f"\nQuality gate PASSED (mAP@50 >= {min_map50:.2f})")
    return eval_results


def export_step(model_path: str, export_format: str,
                export_imgsz: int, half: bool,
                export_simplify: bool, export_opset: int,
                dataset_id: str, eval_results: dict,
                production_tag: str) -> str:
    """Export to ONNX, register in ClearML, and tag the task `production`
    so the model-promote workflow can pick it up. Reaching this step
    implies the quality gate already passed (evaluate_step raises if
    mAP@50 is below threshold)."""
    import hashlib
    from pathlib import Path

    from clearml import Task
    from ultralytics import YOLO

    model = YOLO(model_path)
    onnx_path_str = model.export(
        format=export_format,
        imgsz=export_imgsz,
        half=half,
        simplify=export_simplify,
        opset=export_opset,
    )

    onnx_path = Path(onnx_path_str)
    size_mb = onnx_path.stat().st_size / (1024 * 1024)

    # SHA-256 checksum
    sha256 = hashlib.sha256()
    with open(onnx_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    checksum = sha256.hexdigest()

    print(f"Exported: {onnx_path}")
    print(f"Size:     {size_mb:.2f} MB")
    print(f"SHA-256:  {checksum}")

    # Upload artifact + tag for promotion
    task = Task.current_task()
    if task:
        task.upload_artifact(
            name="fastlane_onnx_model",
            artifact_object=str(onnx_path),
            metadata={
                **eval_results,
                "dataset_id": dataset_id,
                "model_size_mb": round(size_mb, 2),
                "sha256": checksum,
            },
        )
        task.add_tags([production_tag])
        print(f"ONNX uploaded; task tagged `{production_tag}`.")

    return str(onnx_path)


# ═══════════════════════════════════════════════════════════════════════
# Pipeline Controller
# ═══════════════════════════════════════════════════════════════════════

VALID_MODES = ("train", "hpo+train", "hpo-only")


def main():
    parser = argparse.ArgumentParser(description="VoiceEye training pipeline")
    parser.add_argument("--config", default="training/config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--remote", action="store_true",
                        help="Run steps on remote clearml-agent queues")
    parser.add_argument("--mode", default=None, choices=VALID_MODES,
                        help="Override pipeline_mode from config")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pipeline_mode = args.mode or cfg.get("pipeline_mode", "train")
    if pipeline_mode not in VALID_MODES:
        raise ValueError(f"Invalid pipeline_mode: {pipeline_mode!r} "
                         f"(expected one of {VALID_MODES})")

    gpu_queue = cfg.get("queue_gpu", "gpu") if args.remote else None
    cpu_queue = cfg.get("queue_cpu", "cpu") if args.remote else None

    git_sha = os.environ.get("GIT_SHA", "local")[:8]
    reason = os.environ.get("REASON", "manual")[:40]

    pipe = PipelineController(
        name=f"VoiceEye FastLane Pipeline (mode={pipeline_mode} "
             f"sha={git_sha} reason={reason})",
        project=cfg["clearml_project"],
        version="1.0",
        add_pipeline_tags=True,
    )

    # Step 1: Always — Data Preparation (CPU)
    pipe.add_function_step(
        name="data_prep",
        function=data_prep_step,
        function_kwargs={"dataset_id": cfg["dataset_id"]},
        function_return=["dataset_path"],
        execution_queue=cpu_queue,
    )

    # Step 2 (optional): HPO. Controller runs on cpu queue; trials it
    # spawns are dispatched to gpu queue by HyperParameterOptimizer.
    if pipeline_mode in ("hpo+train", "hpo-only"):
        pipe.add_function_step(
            name="hpo",
            function=hpo_step,
            function_kwargs={
                "dataset_path": "${data_prep.dataset_path}",
                "config": cfg,
            },
            function_return=["hpo_params"],
            execution_queue=cpu_queue,
            parents=["data_prep"],
            # ClearML's auto-import detection treats `from clearml.automation.optuna
            # import OptimizerOptuna` as a clearml-only import, so optuna itself
            # never made it into the step's auto-built venv (build failed mid-run
            # with `ModuleNotFoundError: optuna`). Pin it explicitly here.
            packages=["optuna>=3.5.0"],
        )

    # Steps 3-5 (only when mode includes the full train chain).
    if pipeline_mode in ("train", "hpo+train"):
        train_kwargs = {
            "dataset_path": "${data_prep.dataset_path}",
            "config": cfg,
        }
        train_parents = ["data_prep"]
        if pipeline_mode == "hpo+train":
            train_kwargs["hpo_params"] = "${hpo.hpo_params}"
            train_parents = ["hpo"]

        pipe.add_function_step(
            name="train",
            function=train_step,
            function_kwargs=train_kwargs,
            function_return=["model_path", "map50"],
            execution_queue=gpu_queue,
            parents=train_parents,
        )

        pipe.add_function_step(
            name="evaluate",
            function=evaluate_step,
            function_kwargs={
                "model_path": "${train.model_path}",
                "dataset_path": "${data_prep.dataset_path}",
                "conf_threshold": cfg["conf_threshold"],
                "iou_threshold": cfg["iou_threshold"],
                "min_map50": cfg.get("min_map50", 0.40),
            },
            function_return=["eval_results"],
            execution_queue=cpu_queue,
            parents=["train"],
        )

        pipe.add_function_step(
            name="export",
            function=export_step,
            function_kwargs={
                "model_path": "${train.model_path}",
                "export_format": cfg["export_format"],
                "export_imgsz": cfg["export_imgsz"],
                "half": cfg["half"],
                "export_simplify": cfg.get("export_simplify", True),
                "export_opset": cfg.get("export_opset", 17),
                "dataset_id": cfg["dataset_id"],
                "eval_results": "${evaluate.eval_results}",
                "production_tag": cfg.get("production_tag", "production"),
            },
            function_return=["onnx_path"],
            execution_queue=cpu_queue,
            parents=["evaluate"],
        )

    exec_mode = "REMOTE (clearml-agent)" if args.remote else "LOCAL"
    print(f"\nLaunching pipeline ({pipeline_mode}) in {exec_mode} mode...")
    print(f"  GPU queue: {gpu_queue or '(local)'}")
    print(f"  CPU queue: {cpu_queue or '(local)'}")

    if args.remote:
        # Enqueue the controller itself onto a long-lived services queue so
        # it survives the lifetime of the submitter (e.g. a GitHub Actions
        # runner that exits within minutes). The services agent then runs
        # the controller, which dispatches sub-tasks onto gpu/cpu queues.
        services_queue = cfg.get("queue_services", "services")
        print(f"  Controller queue: {services_queue}")
        pipe.start(queue=services_queue)
        print(f"Controller enqueued on '{services_queue}'.")
    else:
        # Local mode: run controller synchronously in this process.
        pipe.start_locally()
    print("Pipeline started. Monitor progress in the ClearML Web UI.")


if __name__ == "__main__":
    main()
