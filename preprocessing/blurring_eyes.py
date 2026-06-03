# =============================================================================
# OVERVIEW
# This script blurs the eyes of a person in a video to protect their identity.
# It uses three layers of detection, in order of accuracy:
#
# 1. MediaPipe FaceMesh  — primary detector, gives precise eye landmark positions
# 2. Haar Cascade        — fallback detector, less precise but better at finding
#                          small/distant faces when FaceMesh fails
# 3. Last known boxes    — static fallback, reuses the last successfully detected
#                          eye positions if both detectors fail
# 4. Black frame         — last resort, blacks out the entire frame if nothing works
#
# MAIN FUNCTIONS:
#   blur_eyes_crop()     — applies blur to eyes using FaceMesh landmarks
#   try_haar_fallback()  — attempts to find eyes using Haar cascade detector
#   process_video()      — main loop that reads frames, detects, blurs, and writes
# =============================================================================

import cv2
import mediapipe as mp
import numpy as np



# Indices of the eye landmarks in MediaPipe's 468-point face mesh
# These trace the contour of each eye
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Haar cascade is built into OpenCV — no download needed
# Better than FaceMesh at detecting small/distant faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# -----------------------------------------------------------------------------
# blur_eyes_crop
# Takes a frame and FaceMesh landmarks, computes a bounding box around each eye,
# and applies a Gaussian blur. Returns the blurred frame and the eye box
# coordinates so they can be reused as a fallback on the next frame.
# -----------------------------------------------------------------------------
def blur_eyes_crop(frame, face_landmarks, det_w, det_h, crop_x1, crop_y1, scale_up):
    eye_boxes = [] # will store (x1, y1, x2, y2) for each eye, used as fallback

    for eye_landmarks in [LEFT_EYE, RIGHT_EYE]:

        # Get the pixel coordinates of each landmark in the upscaled detection frame
        points = np.array([
            (int(face_landmarks.landmark[i].x * det_w),
             int(face_landmarks.landmark[i].y * det_h))
            for i in eye_landmarks
        ], dtype=np.int32)

        # Fit a bounding box around the eye landmark points
        x, y, w, h = cv2.boundingRect(points)

        pad = 15 # extra pixels around eye box — increase if blur clips edges of eyes

        # Convert coordinates from detection frame space back to original frame space:
        # Step 1: divide by scale_up to undo the upscaling
        # Step 2: add crop_x1/crop_y1 to account for the crop offset
        x1 = max(0, int(x / scale_up) + crop_x1 - pad)
        y1 = max(0, int(y / scale_up) + crop_y1 - pad)
        x2 = min(frame.shape[1], int((x + w) / scale_up) + crop_x1 + pad)
        y2 = min(frame.shape[0], int((y + h) / scale_up) + crop_y1 + pad)

        if x2 <= x1 or y2 <= y1: # skip if bounding box is invalid
            continue

        eye_boxes.append((x1, y1, x2, y2)) # save for fallback use

        # Apply Gaussian blur to the eye region
        # (99, 99) = kernel size, must be odd numbers — larger = stronger blur
        # 30 = sigma (spread) — larger = stronger blur
        eye_region = frame[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(eye_region, (99, 99), 30)
        frame[y1:y2, x1:x2] = blurred

    return frame, eye_boxes # return both the blurred frame and the box coordinates

# -----------------------------------------------------------------------------
# try_haar_fallback
# Called when FaceMesh fails to detect a face. Uses OpenCV's Haar cascade
# detector which is better at finding small or distant faces. If a face is
# found, it estimates the eye positions from the face bounding box (eyes are
# roughly in the top half of the face). Returns eye boxes if found, else None.
# -----------------------------------------------------------------------------
def try_haar_fallback(frame, crop_x1, crop_y1, crop_x2, crop_y2, scale_up):
    # Crop and upscale the region of interest, same as the main detection
    """Try to find eyes using Haar cascade when FaceMesh fails.
       Returns eye_boxes if found, None if not."""
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    detection_frame = cv2.resize(crop, (crop.shape[1] * scale_up, crop.shape[0] * scale_up))

    # Haar cascade requires a grayscale image
    gray = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,  # how much to shrink image each pass -- smaller = more thorough but slower
        minNeighbors=3,    # lower = more sensitive, higher = fewer false positives
        minSize=(20, 20)   # minimum face size in pixels — keep small for distant faces
    )

    if len(faces) == 0:
        return None # no face found, signal to caller to use last known boxes

    eye_boxes = []
    for (fx, fy, fw, fh) in faces:
        # Estimate eye positions from the face bounding box:
        # Eyes are in the top half of the face, split into left and right halves
        left_eye_x  = int(fx / scale_up) + crop_x1
        right_eye_x = int((fx + fw // 2) / scale_up) + crop_x1
        eye_y1      = int(fy / scale_up) + crop_y1
        eye_y2      = int((fy + fh // 2) / scale_up) + crop_y1  # top half of face = eye region

        eye_w = int((fw // 2) / scale_up)
        pad = 10

        # Left eye box
        eye_boxes.append((
            max(0, left_eye_x - pad),
            max(0, eye_y1 - pad),
            min(frame.shape[1], left_eye_x + eye_w + pad),
            min(frame.shape[0], eye_y2 + pad)
        ))

        # Right eye box
        eye_boxes.append((
            max(0, right_eye_x - pad),
            max(0, eye_y1 - pad),
            min(frame.shape[1], right_eye_x + eye_w + pad),
            min(frame.shape[0], eye_y2 + pad)
        ))

    return eye_boxes

# -----------------------------------------------------------------------------
# process_video
# Main function. Reads the input video frame by frame, runs detection, applies
# blur, and writes to the output video. Uses a 4-tier fallback system to ensure
# eyes are always blurred even when detection is unreliable.
# -----------------------------------------------------------------------------
def process_video(input_path, output_path):
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(input_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    last_eye_boxes = None # stores eye positions from the last successful detection
    frame_count = 0
    blacked = 0

    with mp_face_mesh.FaceMesh(
        max_num_faces=2, # max faces to detect per frame
        refine_landmarks=True, # enables more accurate eye landmarks (required)
        min_detection_confidence=0.3, # lower = detects more but may get false positives
        min_tracking_confidence=0.3 # lower = keeps tracking longer when face is partially hidden
    ) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # end of video

            frame_count += 1
            h, w = frame.shape[:2]

            # --- Define the crop region where the person appears ---
            # Adjust these if the person is not centered in the frame
            crop_x1 = w // 4 # left edge of crop
            crop_y1 = 1      # top edge of crop
            crop_x2 = 3 * w // 4  # right edge of crop
            crop_y2 = h           # bottom edge of crop
            scale_up = 4          # how much to upscale the crop for detection

            # Crop to the region containing the person, then upscale for detection
            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            detection_frame = cv2.resize(crop, (crop.shape[1] * scale_up, crop.shape[0] * scale_up))


            # JPEG encode/decode cycle — slightly sharpens image in a way that
            # helps MediaPipe detect faces that are small or slightly blurry
            _, buf = cv2.imencode('.jpg', detection_frame)
            detection_frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)

            # Run FaceMesh detection on the upscaled crop
            rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            # --- TIER 1: FaceMesh succeeded ---
            if results.multi_face_landmarks:
                # FaceMesh succeeded — most accurate eye positions
                last_eye_boxes = []
                for face_landmarks in results.multi_face_landmarks:
                    frame, eye_boxes = blur_eyes_crop(frame, face_landmarks,
                                                      detection_frame.shape[1], detection_frame.shape[0],
                                                      crop_x1, crop_y1, scale_up)
                    last_eye_boxes.extend(eye_boxes) # update fallback positions
                out.write(frame)

            else:
                # --- TIER 2: FaceMesh failed — try Haar cascade ---
                # FaceMesh failed — try Haar cascade to get updated eye positions
                haar_boxes = try_haar_fallback(frame, crop_x1, crop_y1, crop_x2, crop_y2, scale_up)

                if haar_boxes is not None:
                    # Haar found the face — use its eye positions and update last_eye_boxes
                    last_eye_boxes = haar_boxes
                    if frame_count % 30 == 0:
                        print(f"Frame {frame_count}: Haar fallback used")

                if last_eye_boxes is not None:
                    # --- TIER 3: blur using best available boxes (Haar or last known FaceMesh) ---
                    # Blur using whatever boxes we have (Haar or last known FaceMesh)
                    for (x1, y1, x2, y2) in last_eye_boxes:
                        eye_region = frame[y1:y2, x1:x2]
                        blurred = cv2.GaussianBlur(eye_region, (99, 99), 30)
                        frame[y1:y2, x1:x2] = blurred
                    out.write(frame)

                else:
                    # --- TIER 4: nothing worked — black out the entire frame ---
                    blacked += 1
                    out.write(np.zeros_like(frame))

            if frame_count % 30 == 0:
                print(f"Frame {frame_count} | Blacked: {blacked}")

    cap.release()
    out.release()
    print(f"Done! Total: {frame_count} | Blacked: {blacked}") # input video path
    print(f"Saved to {output_path}") # output video path

# process_video("arthrogryposis_video.mp4", "output.mp4")

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os


def process_video_wrapper(video_file):
    """
    Worker function executed in a separate process.
    """
    input_path = str(video_file)

    output_dir = Path("processed")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"blurred_{video_file.name}"

    print(f"Starting: {video_file.name}")

    process_video(input_path, str(output_path))

    print(f"Finished: {video_file.name}")

    return video_file.name


if __name__ == "__main__":

    # Find all MP4 files in current directory
    video_folder = Path(r"")
    # video_files = list(Path(".").glob("*.mp4"))
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
                completed_video = future.result()
                print(f"Completed: {completed_video}")
            except Exception as e:
                print(f"Error: {e}")

    print("\nAll videos processed.")