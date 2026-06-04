import cv2
import mediapipe as mp
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time


# =============================================================================
# BLUR HELPERS
# =============================================================================

def blur_face(frame, box, ksize=151, sigma=50):
    """
    Apply a heavy Gaussian blur to a rectangular region of a frame in-place.

    Parameters
    ----------
    frame        : full BGR video frame (numpy array), modified in-place
    box          : (x1, y1, x2, y2) pixel coordinates of the region to blur
    ksize        : kernel size for GaussianBlur — must be odd; larger = blurrier
    sigma        : standard deviation of the Gaussian; larger = blurrier

    The guard clauses prevent crashes when the box is zero-size or outside
    the frame boundaries (can happen at frame edges after padding).
    """
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return  # degenerate box — nothing to blur
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return  # empty slice (can happen at frame edges)
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(region, (ksize, ksize), sigma)


# =============================================================================
# BOX UTILITIES
# =============================================================================

def deduplicate_boxes(boxes, iou_thresh=0.3):
    """
    Remove duplicate/overlapping bounding boxes using Intersection over Union (IoU).

    When FaceDetection returns multiple detections that overlap the same face,
    this function keeps only one. It discards any new box whose IoU with an
    already-kept box exceeds iou_thresh.

    IoU = intersection_area / union_area
      - IoU 1.0 → boxes are identical
      - IoU 0.0 → boxes don't overlap at all
      - iou_thresh=0.3 means "if 30% of the combined area is shared, treat
        as the same face and discard the duplicate"

    Parameters
    ----------
    boxes      : list of (x1, y1, x2, y2) tuples
    iou_thresh : overlap fraction above which a box is considered a duplicate

    Returns
    -------
    Deduplicated list of boxes (order-preserving — first detection wins).
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
            iou     = inter / (area1 + area2 - inter + 1e-6)  # +epsilon avoids /0
            if iou > iou_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return kept


def filter_by_proximity(new_boxes, last_boxes, max_dist=250):
    """
    Discard any detected box whose centre is more than max_dist pixels away
    from every previously-known face position.

    WHY THIS MATTERS
    ----------------
    False positives from FaceDetection tend to appear at random locations —
    a wall texture, piece of equipment, clothing pattern, etc. Real faces
    move smoothly between frames and do not teleport hundreds of pixels
    in a single frame.

    By rejecting boxes implausibly far from the last known face position,
    we eliminate the vast majority of false positives without needing a
    second detector or complex filtering.

    Special case: if last_boxes is None/empty (very first frames before any
    face has been seen), all boxes are accepted unconditionally — there is
    no reference position to compare against yet.

    Parameters
    ----------
    new_boxes : newly detected boxes this frame
    last_boxes: boxes from the most recent reliable detection
    max_dist  : maximum allowed distance (pixels) between box centres
    """
    if not last_boxes:
        return new_boxes  # no prior reference — accept everything
    result = []
    for box in new_boxes:
        cx = (box[0] + box[2]) / 2   # centre-x of candidate box
        cy = (box[1] + box[3]) / 2   # centre-y of candidate box
        for (lx1, ly1, lx2, ly2) in last_boxes:
            lcx = (lx1 + lx2) / 2   # centre-x of last known face
            lcy = (ly1 + ly2) / 2   # centre-y of last known face
            dist = ((cx - lcx) ** 2 + (cy - lcy) ** 2) ** 0.5  # Euclidean distance
            if dist < max_dist:
                result.append(box)
                break  # matched a known face — no need to check others
    return result


def smooth_box(prev, curr, alpha=0.55):
    """
    Exponential Moving Average (EMA) blend between a previous and current box.

    Without smoothing, small jitter in detection output causes the blur
    rectangle to visibly jump around each frame. EMA blending produces a
    smoothly-gliding box that tracks the face without snapping.

    alpha controls how quickly the box responds to new detections:
      - alpha=1.0 → always use the new detection exactly (no smoothing)
      - alpha=0.0 → never move from the original position
      - alpha=0.55 → 55% weight on new detection, 45% on history
                     (slightly conservative for stability on side-view footage
                      where detection confidence fluctuates more)

    Returns a new box tuple with integer pixel coordinates.
    """
    return tuple(int(alpha * c + (1 - alpha) * p) for p, c in zip(prev, curr))


def update_boxes(last_boxes, new_boxes, alpha=0.55):
    """
    Apply EMA smoothing to a list of face boxes frame-to-frame.

    If the number of detected faces changes (e.g. a second person walks in),
    there is no sensible 1-to-1 pairing between old and new boxes, so we
    hard-update instead of blending to avoid mixing up different faces.

    Parameters
    ----------
    last_boxes : list of boxes from the previous frame
    new_boxes  : list of boxes just detected this frame
    alpha      : EMA weight on the new detection (passed to smooth_box)
    """
    if last_boxes and len(new_boxes) == len(last_boxes):
        # Same number of faces — pair by index and blend each box
        return [smooth_box(p, c, alpha) for p, c in zip(last_boxes, new_boxes)]
    # Different face count — hard update, no blending
    return new_boxes


# =============================================================================
# MAIN VIDEO PROCESSING
# =============================================================================

def process_video(input_path, output_path):
    """
    Process a single video: detect the patient's face every frame and blur it.

    Detection pipeline (in priority order):
    ┌──────────────────────────────────────────────────────────────────────┐
    │ 1. MediaPipe FaceDetection → reliable face bbox, handles profiles     │
    │ 2. Reuse last known position → covers momentary detection dropouts    │
    │ 3. Black frame → only if no face was ever found (very start of video) │
    └──────────────────────────────────────────────────────────────────────┘

    WHY ONLY ONE DETECTOR?
    ----------------------
    Earlier versions used FaceMesh + Haar cascades in addition to
    FaceDetection. In practice on side-view clinical footage:
      - FaceMesh hit 0% (designed for frontal faces)
      - Haar cascades produced false-positive blur boxes on background objects
    FaceDetection (model_selection=1) handles both frontal and profile views
    reliably and is the only detector needed.
    """
    print("=" * 60)
    print(f"  Input : {input_path}")
    print(f"  Output: {output_path}")
    print("=" * 60)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("[ERROR] Failed to open video!")
        return

    # Read video metadata for the output writer and progress display
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    duration_s   = total_frames / fps if fps > 0 else 0

    print(f"  Resolution : {width}x{height}")
    print(f"  FPS        : {fps:.2f}")
    print(f"  Frames     : {total_frames}")
    print(f"  Duration   : {duration_s:.1f}s")
    print("-" * 60)

    # Output writer — mp4v codec, same resolution and fps as the input
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    # last_boxes: the most recent face bounding boxes from a successful detection,
    # after EMA smoothing. Used to:
    #   (a) filter new detections by proximity (reject false positives)
    #   (b) blur the face on frames where detection drops out
    last_boxes  = None

    # Counters for the end-of-run summary
    frame_count = 0
    det_hits    = 0   # frames where FaceDetection found the face
    reuse_hits  = 0   # frames that fell back to last known position
    blacked     = 0   # frames written black (no detection ever seen)
    start_time  = time.time()

    with mp.solutions.face_detection.FaceDetection(
        model_selection=1,              # 1 = full-range model (up to ~5m)
                                        # handles both frontal and profile views
                                        # 0 = short-range (<2m), faster but
                                        #     worse on profiles
        min_detection_confidence=0.45   # slightly above default 0.5 to reduce
                                        # false positives; still catches the face
                                        # reliably in clinical footage
    ) as face_detector:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # end of video or unreadable frame

            frame_count += 1
            h, w = frame.shape[:2]

            # FaceDetection requires RGB; OpenCV loads frames as BGR
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ── Progress line every 30 frames ────────────────────────────
            if frame_count % 30 == 0 or frame_count == 1:
                elapsed  = time.time() - start_time
                pct      = (frame_count / total_frames * 100) if total_frames > 0 else 0
                fps_proc = frame_count / elapsed if elapsed > 0 else 0
                eta_s    = (total_frames - frame_count) / fps_proc if fps_proc > 0 else 0
                print(
                    f"  Frame {frame_count:>5}/{total_frames}"
                    f"  [{pct:5.1f}%]"
                    f"  {fps_proc:5.1f} fps"
                    f"  ETA {eta_s:6.1f}s"
                    f"  | detected={det_hits}  reuse={reuse_hits}  black={blacked}"
                )

            # =============================================================
            # CASE 1 — FaceDetection fires
            #
            # The detector returns bounding boxes in relative [0,1] coords.
            # We convert to pixels, add a 15px margin so the blur covers the
            # full face even if the tight detection box clips the edges, then
            # filter out false positives before blurring.
            # =============================================================
            results = face_detector.process(rgb)

            if results.detections:
                new_boxes = []
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box

                    # Scale from relative [0,1] to pixel coordinates
                    # Clamp to frame boundaries so we never index outside the array
                    x1 = max(0, int(bbox.xmin                 * w) - 15)
                    y1 = max(0, int(bbox.ymin                 * h) - 15)
                    x2 = min(w, int((bbox.xmin + bbox.width)  * w) + 15)
                    y2 = min(h, int((bbox.ymin + bbox.height) * h) + 15)
                    new_boxes.append((x1, y1, x2, y2))

                # Remove overlapping detections of the same face, then reject
                # any box that has appeared far from the last known face position
                new_boxes = filter_by_proximity(deduplicate_boxes(new_boxes), last_boxes)

                if new_boxes:
                    # Smooth box positions to prevent frame-to-frame jitter,
                    # then save as the new reference for future frames
                    last_boxes  = update_boxes(last_boxes, new_boxes)
                    det_hits   += 1

                    for box in last_boxes:
                        blur_face(frame, box)

                    out.write(frame)
                    continue

            # =============================================================
            # CASE 2 — No detection this frame: reuse last known position
            #
            # FaceDetection occasionally misses a frame due to motion blur,
            # lighting changes, or an unusually extreme profile angle. The
            # face has not disappeared — the detector just failed. Reusing
            # the last known (smoothed) box keeps the blur in the right place
            # without any visible gap.
            # =============================================================
            if last_boxes is not None:
                reuse_hits += 1
                for box in last_boxes:
                    blur_face(frame, box)
                out.write(frame)
                continue

            # =============================================================
            # CASE 3 — No detection and no history: write a black frame
            #
            # This only triggers at the very start of the video before the
            # first detection has occurred. A black frame is safer than an
            # unblurred frame because it reveals nothing about the patient.
            # In a well-lit clinical recording this should affect at most
            # the first few frames.
            # =============================================================
            blacked += 1
            out.write(np.zeros_like(frame))

    cap.release()
    out.release()

    # ── Final summary ─────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"  DONE in {elapsed:.1f}s")
    print(f"  Total frames  : {frame_count}")
    print(f"  Detected      : {det_hits}  ({det_hits/frame_count*100:.1f}%)")
    print(f"  Reuse         : {reuse_hits}  ({reuse_hits/frame_count*100:.1f}%)")
    print(f"  Blacked       : {blacked}  ({blacked/frame_count*100:.1f}%)")
    print(f"  Saved to      : {output_path}")
    print("=" * 60)


# =============================================================================
# MULTI-VIDEO BATCH PROCESSING
# =============================================================================

def process_video_wrapper(video_file):
    """
    Thin wrapper around process_video for use with ProcessPoolExecutor.
    Saves output to a 'processed/' subfolder with 'blurred_' prefix.
    Returns the filename so the caller can log which videos have finished.
    """
    input_path  = str(video_file)
    output_dir  = Path("processed")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"blurred_{video_file.name}"

    print(f"\n[START] {video_file.name}")
    process_video(input_path, str(output_path))
    print(f"[DONE]  {video_file.name}")
    return video_file.name


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    # ── Single file ──────────────────────────────────────────────────────────
    process_video(r"C:\Users\zuhai\Documents\AI summer research 2026\Clip_1.mp4", "latestOutput.mp4")

