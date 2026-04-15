# ClearML Agent Setup Guide

This guide covers setting up `clearml-agent` to run VoiceEye training tasks remotely. The agent listens on task queues and executes pipeline steps and HPO trials on your GPU machine.

---

## Prerequisites

- Python 3.11+ on the GPU machine
- NVIDIA drivers + CUDA toolkit installed (verify with `nvidia-smi`)
- ClearML server accessible (self-hosted or `app.clear.ml`)
- `clearml-init` already configured (i.e., `~/clearml.conf` exists with API credentials)

---

## 1. Install the Agent

```bash
pip install clearml-agent
```

Or install from the project requirements:

```bash
pip install -r training/requirements.txt
```

---

## 2. Configure the Agent

If you haven't already run `clearml-init`, the agent needs its own config:

```bash
clearml-agent init
```

This creates `~/clearml.conf` with your ClearML server URL and API credentials.

### Recommended agent settings in `~/clearml.conf`

```ini
agent {
    # Python version for task virtual environments
    default_python = "3.11"

    # Package manager
    package_manager {
        type = pip
    }

    # CUDA version (must match your system)
    cuda_version = "12.1"
    cudnn_version = ""

    # Where to store task venvs and git clones
    # default_base_docker = ""  # Uncomment for Docker mode
}
```

---

## 3. Create Queues

In the **ClearML Web UI** (Workers & Queues section), create two queues:

| Queue | Purpose |
|-------|---------|
| `gpu` | Training tasks (Step 2) and HPO trials — requires GPU |
| `cpu` | Data prep (Step 1), evaluation (Step 3), ONNX export (Step 4) |

You can also create queues via the API:

```python
from clearml.backend_api.session.client import APIClient
client = APIClient()
client.queues.create(name="gpu")
client.queues.create(name="cpu")
```

---

## 4. Start the Agent Daemon

### Option A: Single agent listening on both queues (common setup)

```bash
clearml-agent daemon --queue gpu cpu --gpus 0
```

The agent prioritises queues left-to-right — GPU tasks first, then CPU tasks when idle.

### Option B: Separate agents per queue

```bash
# Terminal 1 — GPU agent
clearml-agent daemon --queue gpu --gpus 0

# Terminal 2 — CPU-only agent
clearml-agent daemon --queue cpu --cpu-only
```

### Option C: Docker mode (recommended for reproducibility)

```bash
clearml-agent daemon --queue gpu --gpus 0 --docker
```

In Docker mode, the agent:
- Pulls a base Docker image with CUDA
- Creates a container per task
- Installs `requirements.txt` inside the container
- Runs the task in full isolation

---

## 5. Running Pipelines with the Agent

Once the agent is listening, trigger pipelines with `--remote`:

```bash
# Pipeline steps are enqueued to gpu/cpu queues
python training/pipeline.py --remote
```

The agent picks up each step, creates a virtual environment, installs dependencies, and runs the step function.

---

## 6. Running HPO with the Agent

HPO trials are enqueued to the GPU queue:

```bash
# First: create a template task (short run)
python training/train.py --epochs 5

# Then: launch HPO (uses clearml-agent for trial execution)
python training/hpo.py --template-task-id <TASK_ID> --max-trials 20
```

Each HPO trial is a cloned task with overridden hyperparameters. The agent runs them and reports metrics back to the HPO controller.

---

## 7. Base Weights on Remote Agents

When `clearml-agent` runs a cloned task, it won't have `yolo26n.pt` locally. Two solutions:

### Option A: Upload weights as a ClearML artifact (recommended)

```python
from clearml import Task
task = Task.current_task()
task.upload_artifact("base_weights", "yolo26n.pt")
```

Then in `train.py`, download the artifact if the local file doesn't exist:

```python
from pathlib import Path
if not Path(cfg["model_weights"]).exists():
    # Download from the template task's artifacts
    weights_path = task.artifacts["base_weights"].get_local_copy()
    cfg["model_weights"] = weights_path
```

### Option B: Let Ultralytics auto-download

If using a standard model name like `yolov8n.pt`, Ultralytics downloads it automatically. For custom models like `yolo26n.pt`, Option A is required.

---

## 8. Monitoring

- **ClearML Web UI**: View running agents, queue lengths, task logs in real-time
- **Agent logs**: By default in `~/.clearml/agent.log`
- **Task logs**: Each task's stdout/stderr is captured and viewable in the ClearML UI

### Useful agent commands

```bash
# List running agents
clearml-agent list

# Run in foreground (for debugging)
clearml-agent daemon --queue gpu --gpus 0 --foreground

# Stop gracefully (finishes current task)
clearml-agent daemon --stop
```

---

## Troubleshooting

| Issue | Solution |
|-------|---------|
| Agent can't find CUDA | Verify `nvidia-smi` works; check `cuda_version` in `~/clearml.conf` |
| `yolo26n.pt` not found | Upload as artifact (see Section 7) or place in the git repo |
| Task stuck in "pending" | Check agent is listening on the correct queue |
| Package install fails | Check `requirements.txt` is committed; try Docker mode |
| OOM during training | Reduce `batch` in `config.yaml` (try 8) |
