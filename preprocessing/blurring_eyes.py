import cv2
import mediapipe as mp
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# -----------------------------------------------------------------------------
# Eye landmark indices (MediaPipe FaceMesh)
# -----------------------------------------------------------------------------
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# -----------------------------------------------------------------------------
# Haar face detector (fallback only for FACE, not eyes)
# -----------------------------------------------------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------------------------------------------------------
# Blur helper functions
# -----------------------------------------------------------------------------
def blur_region(frame, x1, y1, x2, y2, ksize=151, sigma=50):
    if x2 <= x1 or y2 <= y1:
        return
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(region, (ksize, ksize), sigma)


def blur_face(frame, box):
    x1, y1, x2, y2 = box
    blur_region(frame, x1, y1, x2, y2)


# -----------------------------------------------------------------------------
# FaceMesh eye + face extraction
# -----------------------------------------------------------------------------
def process_facemesh(face_landmarks, det_w, det_h, crop_x1, crop_y1, scale_up, frame):
    # Full face bounding box
    pts = np.array([
        (int(lm.x * det_w), int(lm.y * det_h))
        for lm in face_landmarks.landmark
    ])

    fx, fy, fw, fh = cv2.boundingRect(pts)

    face_box = (
        max(0, int(fx / scale_up) + crop_x1 - 20),
        max(0, int(fy / scale_up) + crop_y1 - 20),
        min(frame.shape[1], int((fx + fw) / scale_up) + crop_x1 + 20),
        min(frame.shape[0], int((fy + fh) / scale_up) + crop_y1 + 20)
    )

    # Eye boxes
    eye_boxes = []

    for eye in [LEFT_EYE, RIGHT_EYE]:
        points = np.array([
            (int(face_landmarks.landmark[i].x * det_w),
             int(face_landmarks.landmark[i].y * det_h))
            for i in eye
        ])

        x, y, w, h = cv2.boundingRect(points)

        pad = 15

        x1 = max(0, int(x / scale_up) + crop_x1 - pad)
        y1 = max(0, int(y / scale_up) + crop_y1 - pad)
        x2 = min(frame.shape[1], int((x + w) / scale_up) + crop_x1 + pad)
        y2 = min(frame.shape[0], int((y + h) / scale_up) + crop_y1 + pad)

        if x2 > x1 and y2 > y1:
            eye_boxes.append((x1, y1, x2, y2))

    return face_box, eye_boxes


# -----------------------------------------------------------------------------
# Haar fallback → returns FACE boxes only
# -----------------------------------------------------------------------------
def try_haar_fallback(frame, crop_x1, crop_y1, crop_x2, crop_y2, scale_up):
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    detection_frame = cv2.resize(
        crop,
        (crop.shape[1] * scale_up, crop.shape[0] * scale_up)
    )

    gray = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(20, 20)
    )

    face_boxes = []

    for (fx, fy, fw, fh) in faces:
        x1 = max(0, int(fx / scale_up) + crop_x1 - 10)
        y1 = max(0, int(fy / scale_up) + crop_y1 - 10)
        x2 = min(frame.shape[1], int((fx + fw) / scale_up) + crop_x1 + 10)
        y2 = min(frame.shape[0], int((fy + fh) / scale_up) + crop_y1 + 10)

        face_boxes.append((x1, y1, x2, y2))

    return face_boxes


# -----------------------------------------------------------------------------
# MAIN VIDEO PROCESSING
# -----------------------------------------------------------------------------
def process_video(input_path, output_path):
    mp_face_mesh = mp.solutions.face_mesh

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    last_face_boxes = None
    frame_count = 0
    blacked = 0

    with mp_face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            h, w = frame.shape[:2]

            # Crop region
            crop_x1 = w // 4
            crop_y1 = 0
            crop_x2 = 3 * w // 4
            crop_y2 = h
            scale_up = 4

            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            detection_frame = cv2.resize(
                crop,
                (crop.shape[1] * scale_up, crop.shape[0] * scale_up)
            )

            _, buf = cv2.imencode(".jpg", detection_frame)
            detection_frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)

            rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            # -----------------------------------------------------------------
            # CASE 1: FaceMesh SUCCESS
            # -----------------------------------------------------------------
            if results.multi_face_landmarks:

                face_boxes = []

                for face_landmarks in results.multi_face_landmarks:
                    face_box, eye_boxes = process_facemesh(
                        face_landmarks,
                        detection_frame.shape[1],
                        detection_frame.shape[0],
                        crop_x1,
                        crop_y1,
                        scale_up,
                        frame
                    )

                    # If eye detection is unreliable → blur face instead
                    if len(eye_boxes) != 2:
                        blur_face(frame, face_box)
                        face_boxes.append(face_box)
                    else:
                        for box in eye_boxes:
                            blur_region(frame, *box)

                        face_boxes.append(face_box)

                last_face_boxes = face_boxes
                out.write(frame)
                continue

            # -----------------------------------------------------------------
            # CASE 2: FaceMesh FAIL → Haar fallback
            # -----------------------------------------------------------------
            haar_faces = try_haar_fallback(frame, crop_x1, crop_y1, crop_x2, crop_y2, scale_up)

            if haar_faces:
                last_face_boxes = haar_faces

                for box in haar_faces:
                    blur_face(frame, box)

                out.write(frame)
                continue

            # -----------------------------------------------------------------
            # CASE 3: reuse last known face boxes
            # -----------------------------------------------------------------
            if last_face_boxes is not None:
                for box in last_face_boxes:
                    blur_face(frame, box)

                out.write(frame)
                continue

            # -----------------------------------------------------------------
            # CASE 4: nothing detected → black frame
            # -----------------------------------------------------------------
            blacked += 1
            out.write(np.zeros_like(frame))

    cap.release()
    out.release()

    print(f"Done! Frames: {frame_count}, Blacked: {blacked}")
    print(f"Saved to: {output_path}")


# -----------------------------------------------------------------------------
# MULTI VIDEO PROCESSING
# -----------------------------------------------------------------------------
def process_video_wrapper(video_file):
    input_path = str(video_file)

    output_dir = Path("processed")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"blurred_{video_file.name}"

    print(f"Starting: {video_file.name}")
    process_video(input_path, str(output_path))
    print(f"Finished: {video_file.name}")

    return video_file.name


if __name__ == "__main__":

    video_folder = Path("")
    video_files = list(video_folder.glob("*.mp4"))

    if not video_files:
        print("No MP4 files found.")
        exit()

    print(f"Found {len(video_files)} videos.")
    print("Processing up to 7 videos simultaneously...\n")

    with ProcessPoolExecutor(max_workers=7) as executor:
        futures = [
            executor.submit(process_video_wrapper, video)
            for video in video_files
        ]

        for future in as_completed(futures):
            try:
                print(f"Completed: {future.result()}")
            except Exception as e:
                print(f"Error: {e}")

    print("\nAll videos processed.")