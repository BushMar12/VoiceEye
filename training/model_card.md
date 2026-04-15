# VoiceEye YOLO26n Model Card

## Model Details

| Property | Value |
|----------|-------|
| Architecture | YOLO26n (2.57M params, 6.1 GFLOPs) |
| Format | ONNX (opset 17, float32) |
| Input | `[1, 3, 640, 640]` NCHW, normalised [0, 1] |
| Output | `[1, 84, 8400]` — 80 COCO classes |
| Runtime | ONNX Runtime Web (WASM backend) |

## Training

| Property | Value |
|----------|-------|
| Dataset | ClearML ID `{dataset_id}` |
| Base weights | `yolo26n.pt` (COCO pretrained) |
| Epochs | {epochs} |
| Early stopped | {early_stopped} |
| Optimizer | {optimizer} |
| Learning rate | {lr0} (cosine annealing to {lr0} x {lrf}) |
| Batch size | {batch} |
| Image size | 640 x 640 |

## Metrics

| Metric | Value |
|--------|-------|
| mAP@50 | {map50} |
| mAP@50-95 | {map50_95} |
| Precision | {precision} |
| Recall | {recall} |

## Artifact

| Property | Value |
|----------|-------|
| File size | {size_mb} MB |
| SHA-256 | `{checksum}` |
| ClearML Task ID | {task_id} |

## Intended Use

Real-time in-browser object detection for visually impaired users.
Runs at ~10 fps on mobile devices via ONNX Runtime Web WASM backend.

## Limitations

- Trained on COCO classes only; custom/domain-specific objects are not detected.
- Distance estimation uses a pinhole camera model with assumed object heights — accuracy varies.
- Performance degrades in low light and heavy occlusion.
- Float32 only (WASM backend does not support float16).
