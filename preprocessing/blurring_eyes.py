"""
blur_faces.py
─────────────────────────────────────────────────────────────────────────────
Patient face-blurring pipeline for clinical video recordings.

Detection strategy (layered, in priority order):
  1. MediaPipe FaceDetection  — primary detector, handles frontal + profile
  2. OpenCV DNN (SSD ResNet)  — fallback when MediaPipe misses a frame
  3. Last-known position      — covers short dropout bursts (≤ MAX_REUSE_FRAMES)
  4. Black frame              — safest option when nothing else is available

Key improvements over v1:
  • Generous proportional box expansion (35 % of face size per side)
  • Drift expansion — blur box grows slightly for each consecutive missed frame
  • Second detector (OpenCV DNN) eliminates most inter-detector blind spots
  • Faster EMA (alpha 0.75) keeps the blur box on fast-moving faces
  • MAX_REUSE_FRAMES hard cap prevents a stale box from persisting indefinitely

Dependencies:
  pip install opencv-python mediapipe numpy

OpenCV DNN model files (place alongside this script, or update the paths):
  deploy.prototxt
  res10_300x300_ssd_iter_140000.caffemodel
  Download from:
  https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import time


# ─────────────────────────────────────────────────────────────────────────────
# TUNEABLE CONSTANTS  (adjust here rather than hunting through the code)
# ─────────────────────────────────────────────────────────────────────────────

# How far the blur box extends beyond the detected face (fraction of face size).
# 0.35 = 35 % extra on every side — covers hair, ears, and detection jitter.
BOX_PAD_FRAC = 0.35

# Exponential Moving Average weight on the *new* detection.
# Higher → box tracks the face faster; lower → smoother but lags behind.
EMA_ALPHA = 0.75

# Maximum number of consecutive frames we will reuse the last known position
# before giving up and writing a black frame instead.
MAX_REUSE_FRAMES = 15

# Extra pixels added to the blur box per consecutive missed frame, so the box
# expands to chase a moving face during a detection dropout.
DRIFT_PAD_PER_FRAME = 3      # pixels/frame
DRIFT_PAD_MAX       = 40     # cap so the box doesn't grow unboundedly

# IoU threshold above which two boxes are treated as the same face.
IOU_THRESH = 0.30

# Maximum centre-to-centre distance (pixels) between a new box and the
# nearest previously-known face. Boxes beyond this are rejected as false
# positives.
PROXIMITY_MAX_DIST = 250

# MediaPipe FaceDetection settings
MP_MODEL_SELECTION      = 1     # 1 = full-range model (up to ~5 m)
MP_MIN_CONFIDENCE       = 0.45  # slightly below 0.5 to catch difficult angles

# OpenCV DNN FaceDetector settings
DNN_CONF_THRESH         = 0.50  # minimum confidence to accept a DNN detection
DNN_PROTOTXT            = "deploy.prototxt"
DNN_CAFFEMODEL          = "res10_300x300_ssd_iter_140000.caffemodel"

# Gaussian blur parameters for the face region
BLUR_KSIZE = 151   # kernel size — must be odd; larger = blurrier
BLUR_SIGMA = 50    # Gaussian sigma; larger = blurrier


# ─────────────────────────────────────────────────────────────────────────────
# BLUR HELPER
# ─────────────────────────────────────────────────────────────────────────────

def blur_face(frame: np.ndarray, box: tuple) -> None:
    """
    Apply heavy Gaussian blur to a rectangular region of a frame in-place.

    Parameters
    ----------
    frame : full BGR video frame (numpy array), modified in-place
    box   : (x1, y1, x2, y2) pixel coordinates of the region to blur
    """
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(region, (BLUR_KSIZE, BLUR_KSIZE), BLUR_SIGMA)


# ─────────────────────────────────────────────────────────────────────────────
# BOX UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def expand_box(box: tuple, frame_w: int, frame_h: int,
               pad_frac: float = BOX_PAD_FRAC,
               extra_px: int = 0) -> tuple:
    """
    Expand a bounding box outward by a fraction of its own size, plus an
    optional flat pixel amount, clamped to the frame boundaries.

    Parameters
    ----------
    box      : (x1, y1, x2, y2)
    frame_w  : frame width  in pixels
    frame_h  : frame height in pixels
    pad_frac : fractional padding (BOX_PAD_FRAC default)
    extra_px : additional flat padding in pixels (used for drift expansion)
    """
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * pad_frac) + extra_px
    pad_y = int(bh * pad_frac) + extra_px
    return (
        max(0,       x1 - pad_x),
        max(0,       y1 - pad_y),
        min(frame_w, x2 + pad_x),
        min(frame_h, y2 + pad_y),
    )


def deduplicate_boxes(boxes: list, iou_thresh: float = IOU_THRESH) -> list:
    """
    Remove overlapping bounding boxes using Intersection over Union (IoU).
    Keeps the first occurrence; discards subsequent boxes that overlap it
    by more than iou_thresh.
    """
    if not boxes:
        return []
    kept = []
    for box in boxes:
        x1, y1, x2, y2 = box
        duplicate = False
        for kx1, ky1, kx2, ky2 in kept:
            inter_w = max(0, min(x2, kx2) - max(x1, kx1))
            inter_h = max(0, min(y2, ky2) - max(y1, ky1))
            inter   = inter_w * inter_h
            area1   = (x2 - x1) * (y2 - y1)
            area2   = (kx2 - kx1) * (ky2 - ky1)
            iou     = inter / (area1 + area2 - inter + 1e-6)
            if iou > iou_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return kept


def filter_by_proximity(new_boxes: list, last_boxes: list,
                        max_dist: float = PROXIMITY_MAX_DIST) -> list:
    """
    Discard new boxes whose centre is further than max_dist pixels from
    every previously-known face position.

    If last_boxes is None or empty (start of video), all boxes are accepted.
    """
    if not last_boxes:
        return new_boxes
    result = []
    for box in new_boxes:
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        for lx1, ly1, lx2, ly2 in last_boxes:
            dist = ((cx - (lx1 + lx2) / 2) ** 2 +
                    (cy - (ly1 + ly2) / 2) ** 2) ** 0.5
            if dist < max_dist:
                result.append(box)
                break
    return result


def smooth_box(prev: tuple, curr: tuple, alpha: float = EMA_ALPHA) -> tuple:
    """
    Exponential Moving Average blend between a previous and current box.
    alpha=1.0 → use new detection exactly; alpha=0.0 → never move.
    """
    return tuple(int(alpha * c + (1 - alpha) * p) for p, c in zip(prev, curr))


def update_boxes(last_boxes: list, new_boxes: list,
                 alpha: float = EMA_ALPHA) -> list:
    """
    Apply EMA smoothing frame-to-frame.
    If the face count changes, hard-update (no blending) to avoid mixing
    boxes from different faces.
    """
    if last_boxes and len(new_boxes) == len(last_boxes):
        return [smooth_box(p, c, alpha) for p, c in zip(last_boxes, new_boxes)]
    return new_boxes  # hard update


# ─────────────────────────────────────────────────────────────────────────────
# DNN DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

def load_dnn_detector(prototxt: str = DNN_PROTOTXT,
                      caffemodel: str = DNN_CAFFEMODEL):
    """
    Load the OpenCV DNN (SSD ResNet) face detector.
    Returns the net, or None if the model files are not found.
    """
    if not (Path(prototxt).exists() and Path(caffemodel).exists()):
        print(
            f"[WARN] DNN model files not found ({prototxt}, {caffemodel}).\n"
            "       Fallback detector disabled. Download them from:\n"
            "       https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector"
        )
        return None
    net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    print("[INFO] OpenCV DNN fallback detector loaded.")
    return net


def detect_dnn(frame: np.ndarray, net, conf_thresh: float = DNN_CONF_THRESH) -> list:
    """
    Run the OpenCV DNN face detector on a BGR frame.
    Returns a list of raw (x1, y1, x2, y2) pixel boxes (no expansion yet).
    """
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
    )
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < conf_thresh:
            continue
        raw = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = raw.astype(int)
        boxes.append((
            max(0, x1),
            max(0, y1),
            min(w, x2),
            min(h, y2),
        ))
    return boxes


# ─────────────────────────────────────────────────────────────────────────────
# MAIN VIDEO PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_video(input_path: str, output_path: str) -> None:
    """
    Process a single video: detect the patient's face every frame and blur it.

    Detection pipeline (in priority order):
    ┌─────────────────────────────────────────────────────────────────────┐
    │ 1. MediaPipe FaceDetection → reliable face bbox, handles profiles    │
    │ 2. OpenCV DNN SSD ResNet  → fallback when MediaPipe misses           │
    │ 3. Reuse last known pos.  → covers short dropout bursts              │
    │    (box drifts outward by DRIFT_PAD_PER_FRAME px per missed frame)  │
    │ 4. Black frame            → when no reference exists or streak ≥ cap │
    └─────────────────────────────────────────────────────────────────────┘
    """
    print("=" * 65)
    print(f"  Input : {input_path}")
    print(f"  Output: {output_path}")
    print("=" * 65)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("[ERROR] Failed to open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    duration_s   = total_frames / fps if fps > 0 else 0

    print(f"  Resolution : {width}x{height}")
    print(f"  FPS        : {fps:.2f}")
    print(f"  Frames     : {total_frames}")
    print(f"  Duration   : {duration_s:.1f}s")
    print("-" * 65)

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    dnn_net     = load_dnn_detector()
    last_boxes  = None   # smoothed boxes from most recent successful detection
    miss_streak = 0      # consecutive frames without any detection

    frame_count = 0
    det_mp      = 0   # frames detected by MediaPipe
    det_dnn     = 0   # frames detected by DNN fallback
    reuse_hits  = 0   # frames using last-known position
    blacked     = 0   # frames written black
    start_time  = time.time()

    with mp.solutions.face_detection.FaceDetection(
        model_selection=MP_MODEL_SELECTION,
        min_detection_confidence=MP_MIN_CONFIDENCE,
    ) as mp_detector:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            h, w = frame.shape[:2]

            # ── Progress ──────────────────────────────────────────────────
            if frame_count % 30 == 0 or frame_count == 1:
                elapsed  = time.time() - start_time
                pct      = (frame_count / total_frames * 100) if total_frames else 0
                fps_proc = frame_count / elapsed if elapsed else 0
                eta_s    = (total_frames - frame_count) / fps_proc if fps_proc else 0
                print(
                    f"  Frame {frame_count:>5}/{total_frames}"
                    f"  [{pct:5.1f}%]"
                    f"  {fps_proc:5.1f} fps"
                    f"  ETA {eta_s:6.1f}s"
                    f"  | mp={det_mp} dnn={det_dnn}"
                    f"  reuse={reuse_hits} black={blacked}"
                )

            detected_boxes = None   # will hold new raw boxes if detection fires

            # ─────────────────────────────────────────────────────────────
            # CASE 1a — MediaPipe FaceDetection
            # ─────────────────────────────────────────────────────────────
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_detector.process(rgb)

            if results.detections:
                raw_boxes = []
                for detection in results.detections:
                    b  = detection.location_data.relative_bounding_box
                    x1 = max(0, int(b.xmin               * w))
                    y1 = max(0, int(b.ymin               * h))
                    x2 = min(w, int((b.xmin + b.width)   * w))
                    y2 = min(h, int((b.ymin + b.height)  * h))
                    raw_boxes.append((x1, y1, x2, y2))

                raw_boxes = filter_by_proximity(
                    deduplicate_boxes(raw_boxes), last_boxes
                )
                if raw_boxes:
                    detected_boxes = raw_boxes
                    det_mp += 1

            # ─────────────────────────────────────────────────────────────
            # CASE 1b — OpenCV DNN fallback (only if MediaPipe missed)
            # ─────────────────────────────────────────────────────────────
            if detected_boxes is None and dnn_net is not None:
                dnn_boxes = detect_dnn(frame, dnn_net)
                dnn_boxes = filter_by_proximity(
                    deduplicate_boxes(dnn_boxes), last_boxes
                )
                if dnn_boxes:
                    detected_boxes = dnn_boxes
                    det_dnn += 1

            # ─────────────────────────────────────────────────────────────
            # Successful detection — update state and blur
            # ─────────────────────────────────────────────────────────────
            if detected_boxes is not None:
                miss_streak = 0
                last_boxes  = update_boxes(last_boxes, detected_boxes)

                for box in last_boxes:
                    blur_face(frame, expand_box(box, w, h))

                out.write(frame)
                continue

            # ─────────────────────────────────────────────────────────────
            # CASE 2 — No detection: reuse last known position with drift
            # ─────────────────────────────────────────────────────────────
            if last_boxes is not None and miss_streak < MAX_REUSE_FRAMES:
                miss_streak += 1
                drift_px = min(miss_streak * DRIFT_PAD_PER_FRAME, DRIFT_PAD_MAX)

                for box in last_boxes:
                    blur_face(frame, expand_box(box, w, h, extra_px=drift_px))

                reuse_hits += 1
                out.write(frame)
                continue

            # ─────────────────────────────────────────────────────────────
            # CASE 3 — No detection and no usable history → black frame
            # ─────────────────────────────────────────────────────────────
            blacked += 1
            out.write(np.zeros_like(frame))

    cap.release()
    out.release()

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("-" * 65)
    print(f"  DONE in {elapsed:.1f}s")
    print(f"  Total frames      : {frame_count}")
    if frame_count:
        print(f"  MediaPipe hits    : {det_mp:>5}  ({det_mp/frame_count*100:.1f}%)")
        print(f"  DNN fallback hits : {det_dnn:>5}  ({det_dnn/frame_count*100:.1f}%)")
        print(f"  Reuse (drift)     : {reuse_hits:>5}  ({reuse_hits/frame_count*100:.1f}%)")
        print(f"  Blacked           : {blacked:>5}  ({blacked/frame_count*100:.1f}%)")
    print(f"  Saved to          : {output_path}")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_folder(input_dir: str, output_dir: str = "processed") -> None:
    """
    Process every .mp4 in input_dir sequentially and save results to output_dir.
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(input_dir.glob("*.mp4"))
    if not videos:
        print(f"[WARN] No .mp4 files found in {input_dir}")
        return

    print(f"[BATCH] {len(videos)} video(s) to process → {output_dir}")
    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {video.name}")
        out_path = output_dir / f"blurred_{video.name}"
        process_video(str(video), str(out_path))
    print("\n[BATCH] All done.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Single file ──────────────────────────────────────────────────────────
    process_video(
        r"",
        "latestOutput.mp4",
    )

    