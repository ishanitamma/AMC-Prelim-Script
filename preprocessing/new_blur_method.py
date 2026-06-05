import cv2
import mediapipe as mp
import numpy as np
from threading import Thread
from queue import Queue

# ── config ────────────────────────────────────────────────────────────────────
SOURCE       = r""
OUTPUT       = r""
DETECT_EVERY = 5      # re-run FaceDetection every N frames
CROP_PAD     = 0.5   # padding around detected bbox
MASK_DILATE = 30
MESH_SIZE    = 256    # FaceMesh input resolution
BLUR_FACTOR  = 20     # higher = more blur, faster (try 15–25)

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]

# ── helpers ───────────────────────────────────────────────────────────────────
def fast_blur(img, factor=BLUR_FACTOR):
    """Downscale → upscale: same visual result as a huge Gaussian, ~10x faster."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // factor), max(1, h // factor)))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

def soft_blend(frame, blurred, mask):
    alpha = cv2.GaussianBlur(mask, (31, 31), 0)[:, :, None] / 255.0
    return (alpha * blurred + (1 - alpha) * frame).astype(np.uint8)

def rel_bbox_to_pixels(bbox, w, h, pad):
    x1 = max(0, int((bbox.xmin - pad * bbox.width)  * w))
    y1 = max(0, int((bbox.ymin - pad * bbox.height) * h))
    x2 = min(w, int((bbox.xmin + (1 + pad) * bbox.width)  * w))
    y2 = min(h, int((bbox.ymin + (1 + pad) * bbox.height) * h))
    return x1, y1, x2, y2

# ── threaded frame reader (so disk reads don't stall processing) ──────────────
def read_frames(cap, queue):
    while True:
        ret, frame = cap.read()
        queue.put(frame if ret else None)
        if not ret:
            break

# ── main ──────────────────────────────────────────────────────────────────────
cap         = cv2.VideoCapture(SOURCE)
total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps         = cap.get(cv2.CAP_PROP_FPS) or 30
w_out       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_out       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"),
                      fps, (w_out, h_out))

read_queue = Queue(maxsize=64)
Thread(target=read_frames, args=(cap, read_queue), daemon=True).start()

mp_face_detection = mp.solutions.face_detection
mp_face_mesh      = mp.solutions.face_mesh

with (
    mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.3,
    ) as detector,
    mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=4,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as face_mesh,
):
    frame_idx    = 0
    cached_crops = []

    while True:
        frame = read_queue.get()
        if frame is None:
            break

        frame_idx += 1
        h, w = frame.shape[:2]

        # Stage 1 — detection (throttled) ─────────────────────────────────────
        if frame_idx % DETECT_EVERY == 1 or not cached_crops:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            det = detector.process(rgb)
            cached_crops = (
                [rel_bbox_to_pixels(d.location_data.relative_bounding_box,
                                    w, h, CROP_PAD)
                 for d in det.detections]
                if det.detections else []
            )

        if not cached_crops:
            out.write(frame)
            continue

        # Stage 2 — FaceMesh on cropped faces ──────────────────────────────────
        blurred      = fast_blur(frame)          # one blur for the whole frame
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for (x1, y1, x2, y2) in cached_crops:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_h, crop_w = crop.shape[:2]

            crop_rgb = cv2.cvtColor(
                cv2.resize(crop, (MESH_SIZE, MESH_SIZE),
                           interpolation=cv2.INTER_CUBIC),
                cv2.COLOR_BGR2RGB,
            )
            crop_rgb.flags.writeable = False
            mesh = face_mesh.process(crop_rgb)

            if mesh.multi_face_landmarks:
                for lms in mesh.multi_face_landmarks:
                    pts = np.array(
                        [[int(lms.landmark[i].x * crop_w) + x1,
                          int(lms.landmark[i].y * crop_h) + y1]
                         for i in FACE_OVAL],
                        dtype=np.int32,
                    )
                    cv2.fillPoly(combined_mask, [pts], 255)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MASK_DILATE*2+1, MASK_DILATE*2+1))
                    combined_mask = cv2.dilate(combined_mask, kernel)
            else:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.ellipse(combined_mask, (cx, cy),
                            ((x2-x1)//2, (y2-y1)//2), 0, 0, 360, 255, -1)

        out.write(soft_blend(frame, blurred, combined_mask))

        # Progress every 30 frames
        if frame_idx % 30 == 0 or frame_idx == total:
            pct = frame_idx / total * 100 if total else 0
            bar = "█" * int(pct // 2) + "░" * (50 - int(pct // 2))
            print(f"\r[{bar}] {pct:5.1f}%  {frame_idx}/{total} frames", end="", flush=True)

print("\nDone →", OUTPUT)
cap.release()
out.release()
