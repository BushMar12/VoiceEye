"""Decide whether the latest production-tagged ClearML task contains a
model newer than what's currently deployed at public/models/best.onnx.

Outputs `task_id=<id>` to GITHUB_OUTPUT when a newer model exists, or
`task_id=` (empty) when nothing to promote. Prints metadata fields used
by the PR body for human review.

The deployed-path filename `best.onnx` matches src/config.ts'
YOLO_MODEL_PATH so the PWA actually loads what we promote.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from clearml import Task

PROJECT = "VoiceEye"
TAG = "production"
DEPLOYED_ONNX = Path("public/models/best.onnx")


def deployed_sha256() -> str:
    if not DEPLOYED_ONNX.exists():
        return ""
    h = hashlib.sha256()
    with open(DEPLOYED_ONNX, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def latest_production_task() -> Task | None:
    tasks = Task.get_tasks(
        project_name=PROJECT,
        tags=[TAG],
        task_filter={"status": ["completed"], "order_by": ["-last_update"]},
    )
    return tasks[0] if tasks else None


def emit(out_path: str, line: str) -> None:
    """Append a `key=value` line to GITHUB_OUTPUT (or stdout in dev)."""
    if out_path == "-":
        print(line)
    else:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def main() -> None:
    out = os.environ.get("GITHUB_OUTPUT", "-")

    task = latest_production_task()
    if task is None:
        print("No production-tagged task found.")
        emit(out, "task_id=")
        return

    artifact = task.artifacts.get("fastlane_onnx_model")
    if artifact is None:
        print(f"Task {task.id} has no fastlane_onnx_model artifact; skipping.")
        emit(out, "task_id=")
        return

    md = artifact.metadata or {}
    new_sha = str(md.get("sha256", ""))
    cur_sha = deployed_sha256()
    print(f"Latest task: {task.id}")
    print(f"  new sha256: {new_sha}")
    print(f"  cur sha256: {cur_sha or '(no deployed model)'}")

    if new_sha and new_sha == cur_sha:
        print("Already deployed; nothing to promote.")
        emit(out, "task_id=")
        return

    info = {
        "task_id": task.id,
        "sha256": new_sha,
        **{k: str(v) for k, v in md.items()},
    }
    print(json.dumps(info, indent=2))
    emit(out, f"task_id={task.id}")


if __name__ == "__main__":
    main()
