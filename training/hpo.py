"""
VoiceEye FastLane — Hyperparameter Optimization via ClearML + Optuna.

Uses ClearML's HyperParameterOptimizer with Optuna's TPE (Bayesian)
sampler to find optimal training hyperparameters for YOLO26n.

Prerequisites:
    1. clearml-agent running:  clearml-agent daemon --queue gpu --gpus 0
    2. A template task from:   python training/train.py --epochs 5

Usage:
    python training/hpo.py --template-task-id <TASK_ID>
    python training/hpo.py --template-task-id <TASK_ID> --max-trials 30
"""

import argparse

import yaml
from clearml import Task
from clearml.automation import (
    DiscreteParameterRange,
    HyperParameterOptimizer,
    UniformIntegerParameterRange,
    UniformParameterRange,
)
from clearml.automation.optuna import OptimizerOptuna


def main():
    parser = argparse.ArgumentParser(
        description="HPO for VoiceEye YOLO26n training"
    )
    parser.add_argument(
        "--template-task-id", required=True,
        help="ClearML Task ID of a completed train.py run (the template)"
    )
    parser.add_argument(
        "--config", default="training/config.yaml",
        help="Path to config YAML"
    )
    parser.add_argument(
        "--max-trials", type=int, default=None,
        help="Override max number of HPO trials"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    max_trials = args.max_trials or cfg.get("hpo_max_trials", 20)
    time_limit = cfg.get("hpo_time_limit_minutes", 480)
    concurrent = cfg.get("hpo_concurrent_tasks", 2)
    queue_name = cfg.get("queue_gpu", "gpu")

    # ── HPO Controller Task ──────────────────────────────────────────
    task = Task.init(
        project_name=cfg["clearml_project"],
        task_name="HPO — YOLO26n",
        task_type=Task.TaskTypes.optimizer,
    )

    print(f"HPO Configuration:")
    print(f"  Template task:    {args.template_task_id}")
    print(f"  Max trials:       {max_trials}")
    print(f"  Concurrent tasks: {concurrent}")
    print(f"  Time limit:       {time_limit} min")
    print(f"  Execution queue:  {queue_name}")

    # ── Search Space ─────────────────────────────────────────────────
    # Parameter paths use "training_config/<key>" to match
    # task.connect(cfg, name="training_config") in train.py.
    #
    # HPO trials use reduced epochs for speed (relative ranking only).
    # The final production run uses full epochs from config.yaml.

    hyper_parameters = [
        # ── Learning rate (biggest lever for convergence) ──
        UniformParameterRange(
            "training_config/lr0",
            min_value=0.0001,
            max_value=0.01,
            step_size=0.0001,
        ),
        # ── Final LR ratio ──
        UniformParameterRange(
            "training_config/lrf",
            min_value=0.001,
            max_value=0.1,
            step_size=0.001,
        ),
        # ── Optimizer ──
        DiscreteParameterRange(
            "training_config/optimizer",
            values=["SGD", "AdamW", "Adam"],
        ),
        # ── Batch size ──
        DiscreteParameterRange(
            "training_config/batch",
            values=["8", "16", "32"],  # Strings for ClearML serialisation
        ),
        # ── Weight decay ──
        UniformParameterRange(
            "training_config/weight_decay",
            min_value=0.0001,
            max_value=0.001,
            step_size=0.0001,
        ),
        # ── Warmup epochs ──
        UniformIntegerParameterRange(
            "training_config/warmup_epochs",
            min_value=2,
            max_value=10,
            step_size=1,
        ),
        # ── Mosaic augmentation ──
        UniformParameterRange(
            "training_config/mosaic",
            min_value=0.0,
            max_value=1.0,
            step_size=0.1,
        ),
        # ── Scale augmentation ──
        UniformParameterRange(
            "training_config/scale",
            min_value=0.0,
            max_value=0.9,
            step_size=0.1,
        ),
        # ── Mixup ──
        UniformParameterRange(
            "training_config/mixup",
            min_value=0.0,
            max_value=0.3,
            step_size=0.05,
        ),
        # ── Cosine LR schedule ──
        DiscreteParameterRange(
            "training_config/cos_lr",
            values=["True", "False"],  # Strings for ClearML; coerced in train.py
        ),
        # ── Reduced epochs for HPO trials ──
        DiscreteParameterRange(
            "training_config/epochs",
            values=[str(cfg.get("hpo_trial_epochs", 50))],
        ),
        # ── Reduced patience for HPO trials ──
        DiscreteParameterRange(
            "training_config/patience",
            values=[str(cfg.get("hpo_trial_patience", 15))],
        ),
    ]

    # ── Optimiser ────────────────────────────────────────────────────
    # max_iteration_per_job is the early-stopping threshold inside Optuna
    # (one Ultralytics epoch = one ClearML scalar iteration). Required:
    # OptimizerOptuna.__init__ raises TypeError when this is None in
    # clearml >= 2.x. Aligning with hpo_trial_epochs lets each trial run
    # its full epoch budget without the optimiser cutting it short.
    trial_epochs = int(cfg.get("hpo_trial_epochs", 50))

    optimizer = HyperParameterOptimizer(
        base_task_id=args.template_task_id,
        hyper_parameters=hyper_parameters,

        # Objective: maximise mAP@50 reported by train.py
        # train.py calls: task.get_logger().report_scalar("val", "mAP50", ...)
        objective_metric_title="val",
        objective_metric_series="mAP50",
        objective_metric_sign="max",

        # Optuna TPE sampler (Bayesian, sample-efficient)
        optimizer_class=OptimizerOptuna,
        max_iteration_per_job=trial_epochs,

        # Execution
        max_number_of_concurrent_tasks=concurrent,
        total_max_jobs=max_trials,
        execution_queue=queue_name,
    )

    # ── Launch ───────────────────────────────────────────────────────
    print(f"\nStarting HPO with {max_trials} trials...")
    print(f"Monitor progress in the ClearML Web UI.\n")

    optimizer.set_report_period(2)  # Report status every 2 minutes
    optimizer.start()
    optimizer.wait()  # Block until all trials complete

    # ── Extract Best ─────────────────────────────────────────────────
    top_experiments = optimizer.get_top_experiments(top_k=3)

    if not top_experiments:
        print("No completed experiments found.")
        task.close()
        return

    print(f"\n{'='*60}")
    print("Top 3 Experiments:")
    print(f"{'='*60}")
    for i, exp in enumerate(top_experiments):
        scalars = exp.get_last_scalar_metrics()
        map50 = "N/A"
        if "val" in scalars and "mAP50" in scalars["val"]:
            map50 = f"{scalars['val']['mAP50']['last']:.4f}"
        print(f"  #{i+1} Task {exp.id}  mAP@50: {map50}")

    # ── Save Best to Config ──────────────────────────────────────────
    best = top_experiments[0]
    best_params = best.get_parameters()

    print(f"\nBest trial: {best.id}")
    print("Best hyperparameters:")

    # Load original config to merge into
    with open(args.config) as f:
        original_cfg = yaml.safe_load(f)

    updated_keys = []
    for key, value in best_params.items():
        if not key.startswith("training_config/"):
            continue
        param_name = key.split("/", 1)[1]

        # Skip HPO-specific overrides (restore full training values)
        if param_name in ("epochs", "patience"):
            continue

        # Type coercion based on original config
        if param_name in original_cfg:
            orig_type = type(original_cfg[param_name])
            try:
                if orig_type is bool:
                    value = str(value).lower() in ("true", "1", "yes")
                elif orig_type is int:
                    value = int(float(value))
                elif orig_type is float:
                    value = float(value)
            except (ValueError, TypeError):
                pass

        if param_name in original_cfg:
            old_val = original_cfg[param_name]
            original_cfg[param_name] = value
            print(f"  {param_name}: {old_val} -> {value}")
            updated_keys.append(param_name)

    # Write updated config
    with open(args.config, "w") as f:
        yaml.dump(original_cfg, f, default_flow_style=False, sort_keys=False)

    print(f"\nUpdated {len(updated_keys)} parameters in {args.config}")
    print(f"Epochs and patience preserved at {original_cfg['epochs']} / {original_cfg['patience']}")
    print(f"\nNext step: python training/pipeline.py --remote")

    task.close()
    print("HPO complete.")


if __name__ == "__main__":
    main()
