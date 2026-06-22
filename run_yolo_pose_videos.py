"""
Run YOLO pose estimation on a video using Ultralytics.

This script is portable: it does NOT hardcode anyone's personal file paths.
It works on any computer, as long as you tell it which video to use.

Setup (run once in your terminal):
    pip install -U ultralytics

Usage:
    python run_yolo_pose_videos.py path/to/your_video.mp4
    python run_yolo_pose_videos.py path/to/your_video.mp4 --output path/to/output.mp4
    python run_yolo_pose_videos.py path/to/your_video.mp4 --model yolo11s-pose.pt --conf 0.4

If you don't pass an --output, the annotated video is saved next to the input
video, with "_pose" added to the filename.
"""

import os
import sys
import argparse

# Note: "YOLO 26" is not a valid Ultralytics pose model name.
# The closest current/valid Ultralytics pose models are the YOLO11 pose family,
# e.g. yolo11n-pose.pt (smallest/fastest) up to yolo11x-pose.pt (largest/most accurate).
DEFAULT_MODEL_NAME = "yolo11n-pose.pt"
DEFAULT_CONFIDENCE_THRESHOLD = 0.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO pose estimation on a video and save an annotated copy."
    )
    parser.add_argument(
        "input_video",
        help="Path to the input video file (e.g. /Users/you/Videos/clip.mp4)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the annotated output video. Defaults to "
        "'<input_name>_pose.mp4' in the same folder as the input video.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Ultralytics YOLO pose model to use (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold for detections (default: {DEFAULT_CONFIDENCE_THRESHOLD}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_video_path = os.path.abspath(args.input_video)
    model_name = args.model
    confidence_threshold = args.conf

    if args.output:
        output_video_path = os.path.abspath(args.output)
    else:
        base, ext = os.path.splitext(input_video_path)
        output_video_path = f"{base}_pose{ext or '.mp4'}"

    # --- Check input video exists ---
    if not os.path.isfile(input_video_path):
        print(f"ERROR: Input video not found at: {input_video_path}")
        print("Please check the path and try again.")
        sys.exit(1)

    # --- Import ultralytics (with helpful error if missing) ---
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: The 'ultralytics' package is not installed.")
        print("Install it by running:")
        print("    pip install -U ultralytics")
        sys.exit(1)

    # --- Load the model ---
    print(f"Loading model: {model_name} ...")
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"ERROR: Failed to load model '{model_name}'.")
        print(f"Details: {e}")
        sys.exit(1)

    print("Model loaded successfully.")
    print(f"Running pose estimation on: {input_video_path}")
    print("This may take a while depending on video length and your hardware...")

    output_dir = os.path.dirname(output_video_path)
    output_name = os.path.splitext(os.path.basename(output_video_path))[0]

    # --- Run prediction on the video ---
    try:
        results = model.predict(
            source=input_video_path,
            conf=confidence_threshold,
            save=True,
            project=output_dir,
            name=output_name,
            exist_ok=True,
            stream=True,  # process frame by frame, prints progress per frame
        )

        frame_count = 0
        for r in results:
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")

        print(f"Finished processing {frame_count} frames.")

    except Exception as e:
        print("ERROR: Something went wrong while running pose estimation.")
        print(f"Details: {e}")
        sys.exit(1)

    save_dir = os.path.join(output_dir, output_name)
    print("\nDone!")
    print(f"Annotated video was saved by Ultralytics in: {save_dir}")
    print("Look for a .mp4 or .avi file with the same base name as your input video.")


if __name__ == "__main__":
    main()
