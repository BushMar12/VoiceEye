// Average real-world heights (metres) for common YOLO COCO classes.
// Used with the pinhole camera model: distance = (real_h × focal_px) / bbox_h_px
const KNOWN_HEIGHTS_M: Record<string, number> = {
  person: 1.70,
  bicycle: 1.10,
  car: 1.50,
  motorcycle: 1.10,
  airplane: 4.00,
  bus: 3.20,
  train: 4.00,
  truck: 2.80,
  boat: 1.50,
  'traffic light': 0.80,
  'fire hydrant': 0.70,
  'stop sign': 0.90,
  bench: 0.90,
  bird: 0.20,
  cat: 0.30,
  dog: 0.50,
  horse: 1.60,
  cow: 1.40,
  elephant: 3.00,
  bear: 1.20,
  chair: 0.90,
  couch: 0.90,
  bed: 0.60,
  'dining table': 0.75,
  toilet: 0.80,
  tv: 0.60,
  laptop: 0.30,
  bottle: 0.25,
  cup: 0.12,
};

import { DEFAULT_OBJECT_HEIGHT_M, CAMERA_VFOV_DEG } from '../config';

/**
 * Estimates the distance (in metres) from the camera to an object.
 *
 * @param bboxHeightPx  Height of the bounding box in pixels
 * @param videoHeightPx Height of the video frame in pixels
 * @param className     YOLO class name
 * @returns Human-readable distance string, e.g. "~1.2m", ">5m", or "" if data is invalid
 */
export function estimateDistance(
  bboxHeightPx: number,
  videoHeightPx: number,
  className: string
): string {
  if (bboxHeightPx <= 0 || videoHeightPx <= 0) return '';

  const realHeight = KNOWN_HEIGHTS_M[className.toLowerCase()] ?? DEFAULT_OBJECT_HEIGHT_M;
  const focalPx = videoHeightPx / (2 * Math.tan((CAMERA_VFOV_DEG * Math.PI) / 360));
  const distance = (realHeight * focalPx) / bboxHeightPx;

  if (distance > 10) return '>10m';
  if (distance > 5)  return '>5m';
  return `~${distance.toFixed(1)}m`;
}
