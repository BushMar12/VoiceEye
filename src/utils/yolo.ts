import * as ort from 'onnxruntime-web';

// Single-threaded WASM to avoid SharedArrayBuffer / COOP header requirements
ort.env.wasm.numThreads = 1;
// Use absolute path to bypass Vite's "import from public" restriction
ort.env.wasm.wasmPaths = window.location.origin + '/ort-wasm/';

export interface Detection {
  bbox: [number, number, number, number]; // [x, y, w, h] in video pixel coords
  class: string;
  score: number;
}

// YOLOv8 COCO-80 class names
const YOLO_CLASSES: string[] = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
  'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
  'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
  'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
  'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
  'toothbrush',
];

import { YOLO_INPUT_SIZE, YOLO_DEFAULT_CONF, YOLO_IOU_THRESHOLD } from '../config';

// ── Reusable preprocessing resources (lazy-init singleton) ──────────────
let _canvas: OffscreenCanvas | HTMLCanvasElement | null = null;
let _ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null = null;
let _float32: Float32Array | null = null;

function getResources() {
  if (!_canvas) {
    if (typeof OffscreenCanvas !== 'undefined') {
      _canvas = new OffscreenCanvas(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);
    } else {
      const c = document.createElement('canvas');
      c.width = YOLO_INPUT_SIZE;
      c.height = YOLO_INPUT_SIZE;
      _canvas = c;
    }
    _ctx = _canvas.getContext('2d') as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
    _float32 = new Float32Array(3 * YOLO_INPUT_SIZE * YOLO_INPUT_SIZE);
  }
  return { canvas: _canvas, ctx: _ctx!, float32: _float32! };
}

export async function loadYoloModel(path: string): Promise<ort.InferenceSession> {
  return ort.InferenceSession.create(path, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
}

interface PreprocessResult {
  tensor: ort.Tensor;
  scaleX: number;
  scaleY: number;
  padX: number;
  padY: number;
}

function preprocessFrame(video: HTMLVideoElement): PreprocessResult {
  const { ctx, float32 } = getResources();

  // Letterbox resize to preserve aspect ratio
  const vw = video.videoWidth || YOLO_INPUT_SIZE;
  const vh = video.videoHeight || YOLO_INPUT_SIZE;
  const scale = Math.min(YOLO_INPUT_SIZE / vw, YOLO_INPUT_SIZE / vh);
  const nw = Math.round(vw * scale);
  const nh = Math.round(vh * scale);
  const padX = (YOLO_INPUT_SIZE - nw) / 2;
  const padY = (YOLO_INPUT_SIZE - nh) / 2;

  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);
  ctx.drawImage(video, padX, padY, nw, nh);

  const { data } = ctx.getImageData(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);
  const pixels = YOLO_INPUT_SIZE * YOLO_INPUT_SIZE;

  // Convert RGBA → float32 NCHW [1, 3, 640, 640], normalised to [0, 1]
  for (let i = 0; i < pixels; i++) {
    float32[i]              = data[i * 4]     / 255; // R
    float32[pixels + i]     = data[i * 4 + 1] / 255; // G
    float32[2 * pixels + i] = data[i * 4 + 2] / 255; // B
  }

  return {
    tensor: new ort.Tensor('float32', float32, [1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE]),
    scaleX: 1 / scale,
    scaleY: 1 / scale,
    padX,
    padY,
  };
}

function boxIou(a: number[], b: number[]): number {
  const ax2 = a[0] + a[2], ay2 = a[1] + a[3];
  const bx2 = b[0] + b[2], by2 = b[1] + b[3];
  const ix1 = Math.max(a[0], b[0]), iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(ax2, bx2),   iy2 = Math.min(ay2, by2);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  const union = a[2] * a[3] + b[2] * b[3] - inter;
  return union > 0 ? inter / union : 0;
}

function nms(detections: Detection[], iouThreshold: number): Detection[] {
  const sorted = [...detections].sort((a, b) => b.score - a.score);
  const kept: Detection[] = [];
  const suppressed = new Set<number>();

  for (let i = 0; i < sorted.length; i++) {
    if (suppressed.has(i)) continue;
    kept.push(sorted[i]);
    for (let j = i + 1; j < sorted.length; j++) {
      if (sorted[i].class === sorted[j].class &&
          boxIou(sorted[i].bbox, sorted[j].bbox) > iouThreshold) {
        suppressed.add(j);
      }
    }
  }
  return kept;
}

export async function runYolo(
  session: ort.InferenceSession,
  video: HTMLVideoElement,
  confThreshold: number = YOLO_DEFAULT_CONF
): Promise<Detection[]> {
  const { tensor, scaleX, scaleY, padX, padY } = preprocessFrame(video);

  try {
    const results = await session.run({ images: tensor });

    try {
      const outputName = session.outputNames[0];
      const outputTensor = results[outputName];
      const raw = outputTensor.data as Float32Array;
      const shape = outputTensor.dims; // e.g. [1, 84, 8400] or [1, 300, 6]

      const detections: Detection[] = [];

      // ── TYPE 1: End-to-End NMS Output [1, 300, 6] ───────────────────────
      // Format: [x1, y1, x2, y2, score, class]
      if (shape[1] === 300 && shape[2] === 6) {
        for (let i = 0; i < 300; i++) {
          const b = i * 6;
          const score = raw[b + 4];
          if (score < confThreshold) continue;

          const x1 = raw[b + 0];
          const y1 = raw[b + 1];
          const x2 = raw[b + 2];
          const y2 = raw[b + 3];
          const cls = raw[b + 5];

          detections.push({
            bbox: [
              (x1 - padX) * scaleX,
              (y1 - padY) * scaleY,
              (x2 - x1) * scaleX,
              (y2 - y1) * scaleY,
            ],
            class: YOLO_CLASSES[cls] ?? `class_${cls}`,
            score: score,
          });
        }
        return detections; // No manual NMS needed
      }

      // ── TYPE 2: Standard YOLOv8 Output [1, 84, 8400] ────────────────────
      // Format: [cx, cy, w, h, class0, class1, ...] across 8400 anchors
      const numAnchors = shape[2] || 8400;
      const numFields = shape[1] || 84;

      for (let a = 0; a < numAnchors; a++) {
        let maxScore = confThreshold;
        let maxClass = -1;

        for (let c = 4; c < numFields; c++) {
          const score = raw[c * numAnchors + a];
          if (score > maxScore) {
            maxScore = score;
            maxClass = c - 4;
          }
        }
        if (maxClass === -1) continue;

        const cx = raw[0 * numAnchors + a];
        const cy = raw[1 * numAnchors + a];
        const w  = raw[2 * numAnchors + a];
        const h  = raw[3 * numAnchors + a];

        detections.push({
          bbox: [
            ((cx - w / 2) - padX) * scaleX,
            ((cy - h / 2) - padY) * scaleY,
            w * scaleX,
            h * scaleY,
          ],
          class: YOLO_CLASSES[maxClass] ?? 'unknown',
          score: maxScore,
        });
      }

      return nms(detections, YOLO_IOU_THRESHOLD);
    } finally {
      Object.values(results).forEach(t => t.dispose?.());
    }
  } finally {
    tensor.dispose?.();
  }
}
