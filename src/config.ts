// ── VoiceEye Configuration ─────────────────────────────────────────────
// Single source of truth for all tunable constants.

// ── Fast Lane (Detection Loop) ─────────────────────────────────────────
export const INFERENCE_INTERVAL_MS = 100;   // ~10 fps target

// ── Tracker ────────────────────────────────────────────────────────────
export const TRACKER_MAX_AGE_MS = 1000;           // ms without a match before a track is dropped (fps-invariant)
export const TRACKER_EMA_ALPHA = 0.3;             // smoothing factor for velocity estimation (px/ms)
export const TRACKER_HIGH_CONF_THRESHOLD = 0.5;   // score cutoff splitting stage-A / stage-B detection sets
export const TRACKER_COST_THRESHOLD_HIGH = 0.7;   // max assignment cost accepted in stage A (high-conf dets)
export const TRACKER_COST_THRESHOLD_LOW  = 0.5;   // stricter threshold used in stage B (low-conf recovery)
export const TRACKER_CLASS_HISTORY_LEN = 5;       // sliding window length for class-mode voting
export const TRACKER_CONFIRM_HITS = 2;            // consecutive hits needed to promote tentative → confirmed

// ── Proximity Zones ────────────────────────────────────────────────────
export const PROXIMITY_DANGER_THRESHOLD = 0.5;   // screen area fraction
export const PROXIMITY_NEAR_THRESHOLD = 0.25;    // screen area fraction

// ── Haptic Thresholds (screen area fraction) ───────────────────────────
export const HAPTIC_TRIPLE_PULSE_THRESHOLD = 0.6;
export const HAPTIC_SINGLE_PULSE_THRESHOLD = 0.5;

// ── YOLO Model ─────────────────────────────────────────────────────────
export const YOLO_INPUT_SIZE = 640;
export const YOLO_DEFAULT_CONF = 0.5;
export const YOLO_IOU_THRESHOLD = 0.45;
export const YOLO_MODEL_PATH = '/models/best.onnx';

// ── Distance Estimation ────────────────────────────────────────────────
export const DEFAULT_OBJECT_HEIGHT_M = 0.50;
export const CAMERA_VFOV_DEG = 70;
export const MIN_CAMERA_VFOV_DEG = 40;
export const MAX_CAMERA_VFOV_DEG = 120;

// ── VLM (Slow Lane) ───────────────────────────────────────────────────
export const VLM_TIMEOUT_MS = 60_000;       // allow cold Ollama starts without false timeouts

// Resolve the Slow Lane endpoint at build time.
//
// Priority order:
//   1. VITE_OLLAMA_URL — point at a custom Ollama-compatible server
//      (e.g. a Cloudflare Tunnel to a local Ollama). Used by `npm run
//      tunnel` and for private self-hosted setups.
//   2. `vite dev` → /api/ollama — proxied to http://127.0.0.1:11434 by
//      vite.config.ts for zero-setup local development.
//   3. Production build → /api/vlm — Cloudflare Pages Function backed
//      by Workers AI (see functions/api/vlm.ts).
//
// All three endpoints speak the same Ollama-compatible JSON body:
//   POST <endpoint> { model, prompt, images: [b64], options, stream }
// and return `{ response: string }`.
const VLM_BASE_OVERRIDE = import.meta.env.VITE_OLLAMA_URL;
export const VLM_BACKEND = VLM_BASE_OVERRIDE || import.meta.env.DEV ? 'ollama' : 'cloudflare-workers-ai';
export const VLM_ENDPOINT = VLM_BASE_OVERRIDE
  ? `${VLM_BASE_OVERRIDE.replace(/\/$/, '')}/api/generate`
  : import.meta.env.DEV
    ? '/api/ollama/api/generate'
    : '/api/vlm';
export const VLM_MODEL = 'qwen3-vl:2b';
export const VLM_DISPLAY_LABEL = VLM_BACKEND === 'ollama'
  ? 'Qwen3-VL 2B: Deep Context'
  : 'Workers AI: Deep Context';
export const VLM_UNAVAILABLE_MESSAGE = VLM_BACKEND === 'ollama'
  ? 'Scene description unavailable. The local Qwen3-VL engine is not running.'
  : 'Scene description unavailable. The Cloudflare Workers AI backend is not responding.';
export const VLM_IMAGE_QUALITY = 0.85;
export const VLM_IMAGE_MAX_EDGE = 0;        // 0 disables downscaling; preserve camera frame resolution
export const VLM_MAX_TOKENS = 1024;
export const VLM_MAX_WORDS = 40;

// ── Camera ─────────────────────────────────────────────────────────────
export const CAMERA_MAX_RETRIES = 3;
export const CAMERA_RETRY_DELAY_MS = 2000;

// ── Audio ──────────────────────────────────────────────────────────────
export const BEEP_FREQUENCY_HZ = 880;
export const BEEP_DURATION_S = 0.1;
export const BEEP_GAIN = 0.3;
export const QUICK_TTS_RATE = 1.3;
export const QUICK_TTS_VOLUME = 0.8;

// ── Bounding Box Rendering ─────────────────────────────────────────────
export const BBOX_MIN_SCORE = 0.5;          // minimum score to render a bounding box

// ── Inference Metrics ──────────────────────────────────────────────────
export const METRICS_BUFFER_SIZE = 100;
export const METRICS_LOG_INTERVAL = 300;    // log every N frames

// ── UI Messages ────────────────────────────────────────────────────────
export const FULL_INTRO_MESSAGE = "Camera ready. Tap the screen, or say Voice Eye describe, Voice Eye read, or Voice Eye find followed by an object.";
export const SHORT_INTRO_MESSAGE = 'Ready';

// ── Attention Pipeline ─────────────────────────────────────────────────
import type { ProximityZone } from './utils/tracker';

export const HAZARD_TIER: Record<string, 1 | 2 | 3> = {
  car: 1, truck: 1, bus: 1, motorcycle: 1, bicycle: 1, train: 1,
  person: 2, dog: 2, cat: 2, horse: 2,
};
export const DEFAULT_TIER: 1 | 2 | 3 = 3;

export const TIER_WEIGHT: Record<1 | 2 | 3, number> = { 1: 3.0, 2: 1.5, 3: 0.3 };
export const ZONE_WEIGHT: Record<ProximityZone, number> = { danger: 3.0, near: 1.5, safe: 0.5 };
export const EVICTION_AGE_PENALTY = 0.01;
export const APPROACHING_GROWTH_THRESHOLD_ATTN = 0.05;
export const APPROACHING_BOOST = 1.0;

// Fast Lane speech gate. Rendering still shows detected objects, but TTS is
// limited to safety-relevant tracks to reduce alert fatigue.
export const SPEAK_SAFE_TIER1_HAZARDS = true;
export const SPEAK_NEAR_OR_DANGER_OBJECTS = true;

export const CLUSTER_MIN_MEMBERS = 3;
export const CLUSTER_RADIUS_FRAC = 0.15;
export const CLUSTER_MATCH_FRAC = 0.10;
export const CLUSTER_COUNT_DELTA = 2;

export const BUDGET_WINDOW_MS = 2000;
export const COOLDOWN_APPROACHING_MS = 2000;
export const COOLDOWN_SUSTAINED_MS = 3000;
export const SUSTAINED_ENTRY_DELAY_MS = 3000;
export const SUSTAINED_MAX_COUNT = 3;
export const COOLDOWN_GC_MS = 30_000;

export const DEESCALATION_TONE_HZ = 440;
export const DEESCALATION_TONE_S = 0.05;

export const MAX_DETECTIONS_PER_FRAME = 40;
export const MAX_TRACKS = 30;
export const MAX_RENDERED_BOXES = 8;

export const VERBOSITY_K: Record<'quiet' | 'normal' | 'detailed', number> = {
  quiet: 1, normal: 3, detailed: 6,
};
