import cv2
import mediapipe as mp
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

# -----------------------------------------------------------------------------
# Eye landmark indices (MediaPipe FaceMesh)
#
# MediaPipe FaceMesh maps 478 landmarks onto the face. Each landmark has a
# fixed index. These lists are the indices that correspond to the left and
# right eye contours specifically. We use them to draw a bounding box around
# each eye region.
#
# LEFT_EYE  — the eye on the subject's left (appears on the RIGHT of screen
#              in a front-facing camera, but in a side view only one eye
#              will be visible at a time)
# RIGHT_EYE — the eye on the subject's right
# -----------------------------------------------------------------------------
LEFT_EYE  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Group them so we can loop over both eyes with the same logic
EYE_GROUPS = [LEFT_EYE, RIGHT_EYE]


# =============================================================================
# BLUR HELPERS
# =============================================================================

def blur_region(frame, x1, y1, x2, y2, ksize=151, sigma=50):
    """
    Apply a heavy Gaussian blur to a rectangular region of a frame in-place.

    Parameters
    ----------
    frame        : the full BGR video frame (numpy array), modified in-place
    x1, y1       : top-left corner of the region to blur
    x2, y2       : bottom-right corner of the region to blur
    ksize        : kernel size for GaussianBlur — must be odd; larger = blurrier
    sigma        : standard deviation for the Gaussian kernel; larger = blurrier

    The guard clauses at the top prevent crashes when the box is zero-size or
    out of frame bounds.
    """
    if x2 <= x1 or y2 <= y1:
        return  # degenerate box — nothing to blur
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return  # empty slice (can happen at frame edges)
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(region, (ksize, ksize), sigma)


def blur_face(frame, box):
    """
    Blur the entire face bounding box.
    Used as a fallback when eye landmarks cannot be found (e.g. extreme profile).
    Simply unpacks the 4-tuple and delegates to blur_region.
    """
    blur_region(frame, *box)


# =============================================================================
# BOX UTILITIES
# =============================================================================

def deduplicate_boxes(boxes, iou_thresh=0.3):
    """
    Remove duplicate/overlapping bounding boxes using Intersection over Union (IoU).

    When multiple detectors run on the same face (e.g. frontal cascade AND
    profile cascade both fire), they can return very similar boxes for the same
    face. This function keeps only one box per face by discarding any new box
    whose IoU with an already-kept box exceeds iou_thresh.

    IoU = intersection_area / union_area
      - IoU of 1.0 → boxes are identical
      - IoU of 0.0 → boxes don't overlap at all
      - iou_thresh=0.3 means "if 30% of the combined area is shared, treat as same face"

    Parameters
    ----------
    boxes      : list of (x1, y1, x2, y2) tuples
    iou_thresh : overlap threshold above which the new box is considered a duplicate

    Returns
    -------
    List of deduplicated boxes (order-preserving — first detection wins).
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


def filter_by_proximity(new_boxes, last_boxes, max_dist=200):
    """
    Discard any detected box whose centre is more than max_dist pixels away
    from every previously-known face position.

    WHY THIS MATTERS
    ----------------
    False positives from any detector tend to appear at random locations in the
    frame — a wall texture, a piece of equipment, clothing pattern, etc.
    Real faces move smoothly and continuously between frames; they do not
    teleport 500 pixels in one frame.

    By rejecting boxes that are implausibly far from the last known face
    position, we eliminate the vast majority of false positives.

    Special case: if last_boxes is None or empty (i.e. the very first frames
    before any face has been seen), we accept all boxes unconditionally —
    we have no reference to compare against yet.

    Parameters
    ----------
    new_boxes : newly detected boxes this frame
    last_boxes: boxes from the previous reliable detection
    max_dist  : maximum allowed distance (pixels) between box centres
    """
    if not last_boxes:
        return new_boxes  # no prior reference — accept everything on first detection
    result = []
    for box in new_boxes:
        cx = (box[0] + box[2]) / 2   # centre-x of the new box
        cy = (box[1] + box[3]) / 2   # centre-y of the new box
        for (lx1, ly1, lx2, ly2) in last_boxes:
            lcx = (lx1 + lx2) / 2   # centre-x of the last known box
            lcy = (ly1 + ly2) / 2   # centre-y of the last known box
            dist = ((cx - lcx) ** 2 + (cy - lcy) ** 2) ** 0.5  # Euclidean distance
            if dist < max_dist:
                result.append(box)
                break  # matched — no need to check remaining last_boxes
    return result


def smooth_box(prev, curr, alpha=0.6):
    """
    Exponential Moving Average (EMA) blend between a previous and current box.

    Without smoothing, even small jitter in the detector output causes the
    blur rectangle to visibly jump around each frame, which looks unnatural.
    EMA blending produces a smoothly-gliding box.

    alpha controls how quickly the box tracks new detections:
      - alpha=1.0 → always use the new detection exactly (no smoothing)
      - alpha=0.0 → always use the old position (never moves)
      - alpha=0.6 → 60% weight on the new detection, 40% on history (good balance)

    Returns a new box tuple with integer pixel coordinates.
    """
    return tuple(int(alpha * c + (1 - alpha) * p) for p, c in zip(prev, curr))


def update_boxes(last_boxes, new_boxes, alpha=0.6):
    """
    Apply EMA smoothing to a list of boxes frame-to-frame.

    If the number of detected faces changes (e.g. someone walks into frame),
    there is no sensible 1-to-1 pairing between old and new boxes, so we
    hard-update instead of blending.

    Parameters
    ----------
    last_boxes : list of boxes from the previous frame
    new_boxes  : list of boxes just detected this frame
    alpha      : EMA weight on the new detection (passed through to smooth_box)
    """
    if last_boxes and len(new_boxes) == len(last_boxes):
        # Same number of faces — pair up by index and blend each box
        return [smooth_box(p, c, alpha) for p, c in zip(last_boxes, new_boxes)]
    # Different count — just use the new detections as-is
    return new_boxes


# =============================================================================
# FACEMESH HELPERS
# =============================================================================

def extract_eye_and_face_boxes(face_landmarks, frame_w, frame_h,
                                offset_x=0, offset_y=0, scale=1.0):
    """
    Convert raw FaceMesh landmark objects into pixel bounding boxes for the
    eyes and the full face.

    FaceMesh returns landmarks as NORMALISED coordinates in [0, 1] relative
    to the image it was given. If we ran FaceMesh on a CROP (not the full
    frame), we need to:
      1. Multiply by crop dimensions to get crop-space pixel coords
      2. Divide by scale (if we upscaled the crop before processing)
      3. Add the crop's top-left offset to get full-frame pixel coords

    Parameters
    ----------
    face_landmarks : a single FaceMesh face landmark object
    frame_w, frame_h : width/height of the image FaceMesh actually ran on
                       (i.e. the crop, possibly upscaled)
    offset_x, offset_y : top-left corner of the crop in full-frame coordinates
    scale              : upscale factor applied to the crop before FaceMesh

    Returns
    -------
    eye_boxes : list of (x1,y1,x2,y2) — 0, 1, or 2 entries depending on
                which eyes are visible
    face_box  : (x1,y1,x2,y2) bounding the full face with a 20px margin
    """
    # Collect all 478 landmark positions in crop-space pixels
    pts = np.array([
        (int(lm.x * frame_w), int(lm.y * frame_h))
        for lm in face_landmarks.landmark
    ])

    # Bounding rect of all landmarks = full face box in crop-space
    fx, fy, fw, fh = cv2.boundingRect(pts)

    # Map back to full-frame coordinates, add 20px margin for safety
    face_box = (
        max(0,       int(fx / scale) + offset_x - 20),
        max(0,       int(fy / scale) + offset_y - 20),
        int((fx + fw) / scale) + offset_x + 20,
        int((fy + fh) / scale) + offset_y + 20,
    )

    # Build eye boxes from the specific landmark indices for each eye
    eye_boxes = []
    for eye_indices in EYE_GROUPS:
        eye_pts = np.array([
            (int(face_landmarks.landmark[i].x * frame_w),
             int(face_landmarks.landmark[i].y * frame_h))
            for i in eye_indices
        ])
        x, y, w, h = cv2.boundingRect(eye_pts)  # tight rect around eye points
        pad = 15  # padding so the blur covers eyelashes and brow area too
        x1 = max(0, int(x / scale) + offset_x - pad)
        y1 = max(0, int(y / scale) + offset_y - pad)
        x2 = int((x + w) / scale) + offset_x + pad
        y2 = int((y + h) / scale) + offset_y + pad
        if x2 > x1 and y2 > y1:  # only add if the box is valid (non-zero size)
            eye_boxes.append((x1, y1, x2, y2))

    return eye_boxes, face_box


def facemesh_on_crop(face_mesh, frame, face_bbox, pad=40):
    """
    Run FaceMesh on a tight crop around a single detected face.

    WHY CROP INSTEAD OF FULL FRAME?
    --------------------------------
    Running FaceMesh on a full 1920×1080 frame where the face occupies a
    small region often fails, especially for side profiles. The model was
    trained on frontal faces and struggles with:
      - Small face size relative to frame
      - Profile angle (only half the face visible)
      - Low landmark confidence at distance

    By cropping tightly around the face (found by the more robust FaceDetection
    model) and optionally upscaling, we give FaceMesh a much larger, cleaner
    view of just the face, dramatically improving landmark detection success.

    Parameters
    ----------
    face_mesh  : the active MediaPipe FaceMesh context
    frame      : the full BGR video frame
    face_bbox  : (x1,y1,x2,y2) from FaceDetection — the face region to crop
    pad        : extra pixels to expand the crop on each side (gives FaceMesh
                 context at the edges)

    Returns
    -------
    eye_boxes  : list of eye bounding boxes in full-frame coords, or None if
                 FaceMesh found no landmarks in the crop
    face_box   : face bounding box in full-frame coords, or None
    """
    h, w = frame.shape[:2]
    bx1, by1, bx2, by2 = face_bbox

    # Expand crop boundaries, clamped to frame edges
    cx1 = max(0, bx1 - pad)
    cy1 = max(0, by1 - pad)
    cx2 = min(w, bx2 + pad)
    cy2 = min(h, by2 + pad)

    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None, None  # degenerate crop (face bbox was at frame edge)

    # Upscale very small crops so FaceMesh has enough pixels to work with.
    # 200px is the rough minimum width where FaceMesh landmark detection
    # becomes reliable. Below this, accuracy drops sharply.
    crop_h, crop_w = crop.shape[:2]
    scale = 1.0
    if crop_w < 200:
        scale = 200 / crop_w
        crop  = cv2.resize(crop, (int(crop_w * scale), int(crop_h * scale)))

    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    results  = face_mesh.process(rgb_crop)

    if not results.multi_face_landmarks:
        # FaceMesh still couldn't find landmarks even in the tight crop
        # (typical for extreme profiles > ~70° from frontal)
        return None, None

    # Take the first (highest-confidence) face found in the crop
    landmarks = results.multi_face_landmarks[0]

    # Convert landmark coordinates back to full-frame space
    eye_boxes, face_box = extract_eye_and_face_boxes(
        landmarks,
        crop.shape[1], crop.shape[0],  # dimensions of the (possibly upscaled) crop
        offset_x=cx1, offset_y=cy1,    # crop's top-left in full-frame coords
        scale=scale                     # undo the upscaling
    )
    return eye_boxes, face_box


# =============================================================================
# MAIN VIDEO PROCESSING
# =============================================================================

def process_video(input_path, output_path):
    """
    Process a single video file:
      - Detect face position each frame using MediaPipe FaceDetection
      - Attempt to find eye landmarks using FaceMesh on the detected face crop
      - Blur eyes if found, otherwise blur the full face box
      - Reuse last known position on frames where detection fails
      - Write a black frame only if no face was ever found

    DETECTION PIPELINE (in order of preference):
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. FaceDetection  → finds face bbox (reliable, handles profiles) │
    │ 2. FaceMesh crop  → finds eye landmarks within that bbox         │
    │    ├─ eyes found  → blur eyes only               ← PREFERRED     │
    │    └─ no eyes     → blur full face bbox          ← FALLBACK      │
    │ 3. No detection   → reuse last reliable position ← DROPOUT GUARD │
    │ 4. Nothing ever   → write black frame            ← LAST RESORT   │
    └─────────────────────────────────────────────────────────────────┘
    """
    print("=" * 60)
    print(f"  Input : {input_path}")
    print(f"  Output: {output_path}")
    print("=" * 60)

    mp_face_mesh      = mp.solutions.face_mesh
    mp_face_detection = mp.solutions.face_detection

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("[ERROR] Failed to open video!")
        return

    # Read video metadata
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    duration_s   = total_frames / fps if fps > 0 else 0

    print(f"  Resolution : {width}x{height}")
    print(f"  FPS        : {fps:.2f}")
    print(f"  Frames     : {total_frames}")
    print(f"  Duration   : {duration_s:.1f}s")
    print(f"  Strategy   : eyes preferred → face fallback → reuse → black")
    print("-" * 60)

    # Output writer — uses mp4v codec, same resolution and fps as input
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    # ── State tracking across frames ─────────────────────────────────────────
    # last_reliable_boxes: the most recent face bounding boxes from a trusted
    #   detector (FaceDetection or FaceMesh). Used to filter new detections by
    #   proximity and to fall back to when detection drops out.
    last_reliable_boxes = None

    # last_eye_boxes: the most recent eye bounding boxes (when FaceMesh succeeded).
    #   Kept separately so that during a dropout we can reuse eye-blur rather
    #   than downgrading to face-blur.
    last_eye_boxes      = None

    # last_used_eyes: flag indicating whether the previous detection used eye blur
    #   (True) or face blur (False). Determines reuse behaviour during dropouts.
    last_used_eyes      = False

    # ── Counters for the progress report ─────────────────────────────────────
    frame_count     = 0
    blacked         = 0   # frames written as black (no detection ever)
    eye_hits        = 0   # frames where eye landmarks were found and blurred
    face_hits       = 0   # frames where only face box was available
    reuse_eye_hits  = 0   # dropout frames reusing last eye boxes
    reuse_face_hits = 0   # dropout frames reusing last face boxes
    start_time      = time.time()

    # ── Initialise both MediaPipe models ─────────────────────────────────────
    with mp_face_mesh.FaceMesh(
        max_num_faces=2,           # support up to 2 patients/people in frame
        refine_landmarks=True,     # enables the 478-point model (vs 468-point)
                                   # refine_landmarks adds iris landmarks which
                                   # slightly improves eye region accuracy
        min_detection_confidence=0.3,  # lower threshold = more detections,
        min_tracking_confidence=0.3    # but we filter false positives ourselves
    ) as face_mesh, \
    mp_face_detection.FaceDetection(
        model_selection=1,             # model 1 = full-range (up to 5m distance)
                                       # model 0 = short-range (< 2m), faster
                                       # model 1 handles profile views better
        min_detection_confidence=0.4   # slightly higher than mesh to reduce noise
    ) as face_detector:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # end of video or read error

            frame_count += 1
            h, w = frame.shape[:2]

            # FaceMesh and FaceDetection both require RGB input (OpenCV uses BGR)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ── Progress line every 30 frames ────────────────────────────────
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
                    f"  | eyes={eye_hits}  face={face_hits}"
                    f"  reuse_eye={reuse_eye_hits}  reuse_face={reuse_face_hits}"
                    f"  black={blacked}"
                )

            # =================================================================
            # STEP 1 — FaceDetection: locate the face region this frame
            #
            # FaceDetection is a lightweight model optimised for finding faces
            # at various angles including side profiles. It returns bounding
            # boxes only — no landmarks, no eye positions.
            #
            # We run this on the full frame first to get a rough face location,
            # then pass that location to FaceMesh (Step 2) for fine detail.
            # =================================================================
            det_results = face_detector.process(rgb)
            face_bboxes = []

            if det_results.detections:
                for detection in det_results.detections:
                    # Landmarks are in relative [0,1] coordinates — scale to pixels
                    bbox = detection.location_data.relative_bounding_box
                    x1   = max(0, int(bbox.xmin                 * w) - 10)
                    y1   = max(0, int(bbox.ymin                 * h) - 10)
                    x2   = min(w, int((bbox.xmin + bbox.width)  * w) + 10)
                    y2   = min(h, int((bbox.ymin + bbox.height) * h) + 10)
                    # The -10/+10 gives a small margin around the tight detection box
                    face_bboxes.append((x1, y1, x2, y2))

                # Remove duplicate boxes (in case two detections overlap the same face)
                # then reject any box too far from where we last saw the face
                face_bboxes = filter_by_proximity(
                    deduplicate_boxes(face_bboxes), last_reliable_boxes
                )

            # =================================================================
            # STEP 2 — FaceMesh on crop: try to get eye landmarks
            #
            # For each face found by FaceDetection, we run FaceMesh on just
            # the cropped face region. This is the key improvement over running
            # FaceMesh on the full frame:
            #   - Face is larger relative to the input → better feature detection
            #   - Less background clutter for the model to ignore
            #   - Small crops are upscaled to ≥200px wide before processing
            #
            # Outcome per face:
            #   eye_boxes returned → blur eyes only (preferred)
            #   no eye_boxes       → blur the face bbox from Step 1 (fallback)
            # =================================================================
            if face_bboxes:
                any_eyes_found = False
                current_face_boxes = []  # face boxes confirmed this frame
                current_eye_boxes  = []  # eye boxes found this frame (may be empty)

                for face_bbox in face_bboxes:
                    eye_boxes, face_box = facemesh_on_crop(face_mesh, frame, face_bbox)

                    if eye_boxes:
                        # FaceMesh succeeded — blur only the eye region(s)
                        # On a side view only one eye will be visible, so
                        # eye_boxes will typically have just 1 entry here
                        for box in eye_boxes:
                            blur_region(frame, *box)
                        current_eye_boxes.extend(eye_boxes)
                        current_face_boxes.append(face_bbox)
                        any_eyes_found = True
                    else:
                        # FaceMesh failed (extreme profile, occlusion, etc.)
                        # Fall back to blurring the full face bounding box
                        blur_face(frame, face_bbox)
                        current_face_boxes.append(face_bbox)

                # Update the reliable face position tracker with EMA smoothing
                # This is used for proximity filtering and reuse in future frames
                last_reliable_boxes = update_boxes(last_reliable_boxes, current_face_boxes)

                if any_eyes_found:
                    # At least one face had eye landmarks — save for reuse
                    last_eye_boxes = current_eye_boxes
                    last_used_eyes = True
                    eye_hits      += 1
                else:
                    # No eye landmarks found for any face this frame
                    last_used_eyes = False
                    face_hits     += 1

                out.write(frame)
                continue

            # =================================================================
            # STEP 3 — Reuse last reliable state (detection dropout)
            #
            # FaceDetection occasionally misses a frame — motion blur, unusual
            # lighting, extreme angle, or just model uncertainty. The face is
            # almost certainly still in the same place, so reusing the last
            # known position is safer than leaving the frame unblurred.
            #
            # We prefer to reuse eye boxes if the last successful frame had
            # them, to maintain the more precise eye-only blur. If the last
            # frame only had a face box, we reuse that instead.
            # =================================================================
            if last_reliable_boxes is not None:
                if last_used_eyes and last_eye_boxes:
                    # Last frame found eyes — reuse those eye positions
                    for box in last_eye_boxes:
                        blur_region(frame, *box)
                    reuse_eye_hits += 1
                else:
                    # Last frame only had face box — reuse that
                    for box in last_reliable_boxes:
                        blur_face(frame, box)
                    reuse_face_hits += 1
                out.write(frame)
                continue

            # =================================================================
            # STEP 4 — Black frame (no detection ever seen)
            #
            # This only triggers in the very first frames of the video before
            # any face has been detected. Writing a black frame is preferable
            # to writing an unblurred frame, as that would expose the patient.
            # In a well-lit clinical video this should almost never happen.
            # =================================================================
            blacked += 1
            out.write(np.zeros_like(frame))

    cap.release()
    out.release()

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed       = time.time() - start_time
    total_blurred = eye_hits + face_hits + reuse_eye_hits + reuse_face_hits
    print("-" * 60)
    print(f"  DONE in {elapsed:.1f}s")
    print(f"  Total frames    : {frame_count}")
    print(f"  Eye blur        : {eye_hits}  ({eye_hits/frame_count*100:.1f}%)  ← preferred")
    print(f"  Face blur       : {face_hits}  ({face_hits/frame_count*100:.1f}%)  ← fallback")
    print(f"  Reuse (eyes)    : {reuse_eye_hits}  ({reuse_eye_hits/frame_count*100:.1f}%)")
    print(f"  Reuse (face)    : {reuse_face_hits}  ({reuse_face_hits/frame_count*100:.1f}%)")
    print(f"  Blacked frames  : {blacked}  ({blacked/frame_count*100:.1f}%)")
    print(f"  Total blurred   : {total_blurred}  ({total_blurred/frame_count*100:.1f}%)")
    print(f"  Saved to        : {output_path}")
    print("=" * 60)


# =============================================================================
# MULTI-VIDEO PROCESSING (batch mode)
# =============================================================================

def process_video_wrapper(video_file):
    """
    Wrapper for use with ProcessPoolExecutor in batch mode.
    Each call processes one video file and saves it to the 'processed/' folder.
    Returns the filename so the caller can log completion.
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
    process_video(r"C:\Users\zuhai\Documents\AI summer research 2026\sideview.mp4", "latestOutput.mp4")

