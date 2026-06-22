"""
Run YOLO pose estimation frame-by-frame on a folder of images using Ultralytics.

Setup (run once in your terminal):
    pip install -U ultralytics

Then run this script (use the Python that has ultralytics installed, e.g.):
    /usr/local/bin/python3.10 run_yolo_pose.py
"""

import os
import sys
import glob

# -----------------------------
# CONFIGURATION - edit these
# -----------------------------
INPUT_FRAMES_DIR = "/Users/archishman/Arthrogryposis/Sai"
OUTPUT_FRAMES_DIR = "/Users/archishman/Arthrogryposis/Yolo Pose Test/Sai_pose_output"

# Note: "YOLO 26" is not a valid Ultralytics pose model name.
# The closest current/valid Ultralytics pose models are the YOLO11 pose family,
# e.g. yolo11n-pose.pt (smallest/fastest) up to yolo11x-pose.pt (largest/most accurate).
MODEL_NAME = "yolo11n-pose.pt"

CONFIDENCE_THRESHOLD = 0.5

# Which image extensions to process
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def main():
    # --- Check input folder exists ---
    if not os.path.isdir(INPUT_FRAMES_DIR):
        print(f"ERROR: Input frames folder not found at: {INPUT_FRAMES_DIR}")
        print("Please check the path and try again.")
        sys.exit(1)

    # --- Find frame images ---
    frame_paths = sorted(
        p for p in glob.glob(os.path.join(INPUT_FRAMES_DIR, "*"))
        if p.lower().endswith(IMAGE_EXTENSIONS)
    )

    if not frame_paths:
        print(f"ERROR: No image files found in: {INPUT_FRAMES_DIR}")
        sys.exit(1)

    print(f"Found {len(frame_paths)} frames in: {INPUT_FRAMES_DIR}")

    # --- Import ultralytics (with helpful error if missing) ---
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: The 'ultralytics' package is not installed.")
        print("Install it by running:")
        print("    pip install -U ultralytics")
        sys.exit(1)

    # --- Load the model ---
    print(f"Loading model: {MODEL_NAME} ...")
    try:
        model = YOLO(MODEL_NAME)
    except Exception as e:
        print(f"ERROR: Failed to load model '{MODEL_NAME}'.")
        print(f"Details: {e}")
        sys.exit(1)

    print("Model loaded successfully.")

    # --- Make sure output folder exists ---
    os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)

    print(f"Running pose estimation on {len(frame_paths)} frames...")
    print("This may take a while depending on number of frames and your hardware...")

    # --- Process each frame individually ---
    processed = 0
    failed = 0
    for i, frame_path in enumerate(frame_paths, start=1):
        try:
            results = model.predict(
                source=frame_path,
                conf=CONFIDENCE_THRESHOLD,
                save=False,
                verbose=False,
            )
            annotated = results[0].plot()  # numpy array (BGR) with keypoints/skeleton drawn

            import cv2
            out_path = os.path.join(OUTPUT_FRAMES_DIR, os.path.basename(frame_path))
            cv2.imwrite(out_path, annotated)

            processed += 1
        except Exception as e:
            failed += 1
            print(f"WARNING: Failed to process frame '{frame_path}': {e}")

        if i % 20 == 0 or i == len(frame_paths):
            print(f"Processed {i}/{len(frame_paths)} frames...")

    print("\nDone!")
    print(f"Successfully processed: {processed} frames")
    if failed:
        print(f"Failed: {failed} frames")
    print(f"Annotated frames saved to: {OUTPUT_FRAMES_DIR}")


if __name__ == "__main__":
    main()
