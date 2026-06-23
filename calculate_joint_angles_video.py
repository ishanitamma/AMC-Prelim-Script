
"""
Usage:
    python calculate_joint_angles_video.py path/to/input_video.mp4
    python calculate_joint_angles_video.py path/to/input_video.mp4 --output-video output.mp4 --output-csv angles.csv
    python calculate_joint_angles_video.py path/to/input_video.mp4 --model yolo11s-pose.pt --conf 0.4
    python calculate_joint_angles_video.py path/to/input_video.mp4 --track
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict, deque


DEFAULT_MODEL_NAME = "yolo11n-pose.pt"
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
MIN_KEYPOINT_CONFIDENCE = 0.3
DEFAULT_TRACE_LENGTH = 120

# COCO 17-keypoint indices used by Ultralytics pose models
KP = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# Each angle is the angle at the "vertex" joint, formed by the two other points.
ANGLES_TO_COMPUTE = {
    "left_elbow_angle": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow_angle": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder_angle": ("left_elbow", "left_shoulder", "left_hip"),
    "right_shoulder_angle": ("right_elbow", "right_shoulder", "right_hip"),
    "left_hip_angle": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip_angle": ("right_shoulder", "right_hip", "right_knee"),
    "left_knee_angle": ("left_hip", "left_knee", "left_ankle"),
    "right_knee_angle": ("right_hip", "right_knee", "right_ankle"),
}

ANGLE_COLORS = {
    "left_elbow_angle": (80, 220, 255),
    "right_elbow_angle": (80, 160, 255),
    "left_shoulder_angle": (120, 255, 120),
    "right_shoulder_angle": (60, 210, 90),
    "left_hip_angle": (255, 190, 90),
    "right_hip_angle": (255, 140, 40),
    "left_knee_angle": (220, 130, 255),
    "right_knee_angle": (170, 90, 255),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO Pose on a video, overlay joint angles, and save a CSV."
    )
    parser.add_argument("input_video", help="Path to the input video file.")
    parser.add_argument(
        "--output-video",
        default=None,
        help="Path for the annotated video. Defaults to '<input_name>_joint_angles.mp4'.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Path for the CSV. Defaults to '<input_name>_joint_angles.csv'.",
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
        help=f"Detection confidence threshold (default: {DEFAULT_CONFIDENCE_THRESHOLD}).",
    )
    parser.add_argument(
        "--kp-conf",
        type=float,
        default=MIN_KEYPOINT_CONFIDENCE,
        help=f"Minimum keypoint confidence for angle calculation (default: {MIN_KEYPOINT_CONFIDENCE}).",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Use YOLO tracking IDs when available instead of only per-frame person indexes.",
    )
    parser.add_argument(
        "--trace-length",
        type=int,
        default=DEFAULT_TRACE_LENGTH,
        help=f"Number of recent frames to show in the angle trace panel (default: {DEFAULT_TRACE_LENGTH}).",
    )
    parser.add_argument(
        "--no-traces",
        action="store_true",
        help="Do not draw the live-updating angle trace panel.",
    )
    parser.add_argument(
        "--max-people",
        type=int,
        default=None,
        help="Optional maximum number of detected people to write/draw per frame.",
    )
    return parser.parse_args()


def default_output_paths(input_video_path):
    base, _ = os.path.splitext(input_video_path)
    return f"{base}_joint_angles.mp4", f"{base}_joint_angles.csv"


def calculate_angle(a, b, c):
    """Angle at point b, formed by rays b->a and b->c, in degrees."""
    ax, ay = a
    bx, by = b
    cx, cy = c

    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - by)

    v1_len = math.hypot(*v1)
    v2_len = math.hypot(*v2)
    if v1_len == 0 or v2_len == 0:
        return None

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cos_angle = max(-1.0, min(1.0, dot / (v1_len * v2_len)))
    return math.degrees(math.acos(cos_angle))


def make_get_point(xy, conf, person_idx, min_keypoint_conf):
    def get_point(name):
        idx = KP[name]
        point_conf = conf[person_idx][idx] if conf is not None else 1.0
        if point_conf < min_keypoint_conf:
            return None
        x, y = xy[person_idx][idx]
        if x == 0 and y == 0:
            return None
        return (float(x), float(y))

    return get_point


def compute_person_angles(xy, conf, person_idx, min_keypoint_conf):
    get_point = make_get_point(xy, conf, person_idx, min_keypoint_conf)
    angles = {}
    vertices = {}

    for angle_name, (p1_name, vertex_name, p2_name) in ANGLES_TO_COMPUTE.items():
        p1 = get_point(p1_name)
        vertex = get_point(vertex_name)
        p2 = get_point(p2_name)
        vertices[angle_name] = vertex

        if p1 is None or vertex is None or p2 is None:
            angles[angle_name] = None
            continue

        angle = calculate_angle(p1, vertex, p2)
        angles[angle_name] = round(angle, 1) if angle is not None else None

    return angles, vertices


def get_person_id(result, person_idx):
    if result.boxes is None or result.boxes.id is None:
        return f"person_{person_idx}"

    track_ids = result.boxes.id.cpu().numpy()
    if person_idx >= len(track_ids):
        return f"person_{person_idx}"

    return str(int(track_ids[person_idx]))


def draw_text_with_background(frame, text, origin, color, scale=0.5, thickness=1):
    import cv2

    x, y = int(origin[0]), int(origin[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 4
    top_left = (max(0, x - pad), max(0, y - text_h - baseline - pad))
    bottom_right = (
        min(frame.shape[1] - 1, x + text_w + pad),
        min(frame.shape[0] - 1, y + baseline + pad),
    )
    cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_angle_labels(frame, person_id, angles, vertices):
    for angle_name, angle in angles.items():
        if angle is None or vertices[angle_name] is None:
            continue

        vx, vy = vertices[angle_name]
        color = ANGLE_COLORS.get(angle_name, (255, 255, 255))
        short_name = angle_name.replace("_angle", "").replace("_", " ")
        text = f"{short_name}: {angle:.1f}"
        if person_id is not None:
            text = f"{person_id} {text}"
        draw_text_with_background(frame, text, (vx + 8, vy - 8), color)


def update_traces(trace_history, person_id, angles, trace_length):
    person_traces = trace_history[person_id]
    for angle_name in ANGLES_TO_COMPUTE:
        if angle_name not in person_traces:
            person_traces[angle_name] = deque(maxlen=trace_length)
        person_traces[angle_name].append(angles.get(angle_name))


def draw_trace_panel(frame, trace_history, active_person_id):
    import cv2

    if active_person_id not in trace_history:
        return

    height, width = frame.shape[:2]
    panel_w = min(360, max(260, width // 3))
    panel_h = min(260, max(180, height // 3))
    x0 = width - panel_w - 12
    y0 = 12
    x1 = width - 12
    y1 = y0 + panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (210, 210, 210), 1)

    title = f"Angle traces: {active_person_id}"
    cv2.putText(
        frame,
        title,
        (x0 + 10, y0 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    plot_x0 = x0 + 108
    plot_x1 = x1 - 10
    row_h = max(22, (panel_h - 38) // len(ANGLES_TO_COMPUTE))
    trace_area_w = max(1, plot_x1 - plot_x0)

    for row_idx, angle_name in enumerate(ANGLES_TO_COMPUTE):
        row_y_mid = y0 + 42 + row_idx * row_h
        color = ANGLE_COLORS.get(angle_name, (255, 255, 255))
        label = angle_name.replace("_angle", "").replace("_", " ")
        history = list(trace_history[active_person_id].get(angle_name, []))
        latest = next((v for v in reversed(history) if v is not None), None)
        latest_text = "" if latest is None else f"{latest:.1f}"

        cv2.putText(
            frame,
            label[:15],
            (x0 + 10, row_y_mid + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            latest_text,
            (x0 + 74, row_y_mid + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )

        cv2.line(frame, (plot_x0, row_y_mid), (plot_x1, row_y_mid), (75, 75, 75), 1)
        points = []
        for i, value in enumerate(history[-trace_area_w:]):
            if value is None:
                points.append(None)
                continue
            x = plot_x0 + i
            normalized = max(0.0, min(1.0, value / 180.0))
            y = int(row_y_mid + row_h * 0.42 - normalized * row_h * 0.84)
            points.append((x, y))

        previous = None
        for point in points:
            if point is not None and previous is not None:
                cv2.line(frame, previous, point, color, 1)
            previous = point


def open_video_writer(output_video_path, fps, frame_size):
    import cv2

    os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
    ext = os.path.splitext(output_video_path)[1].lower()
    fourcc_name = "XVID" if ext == ".avi" else "mp4v"
    fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {output_video_path}")
    return writer


def write_csv(output_csv_path, rows):
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    fieldnames = [
        "frame_number",
        "timestamp_seconds",
        "person_id",
        "person_index",
    ] + list(ANGLES_TO_COMPUTE.keys())

    with open(output_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    input_video_path = os.path.abspath(args.input_video)

    default_video_path, default_csv_path = default_output_paths(input_video_path)
    output_video_path = os.path.abspath(args.output_video or default_video_path)
    output_csv_path = os.path.abspath(args.output_csv or default_csv_path)

    if not os.path.isfile(input_video_path):
        print(f"ERROR: Input video not found at: {input_video_path}")
        sys.exit(1)

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as e:
        print(f"ERROR: Missing required package: {e.name}")
        print("Install dependencies by running:")
        print("    pip install -U ultralytics")
        print("OpenCV is installed automatically with ultralytics in most setups.")
        sys.exit(1)

    capture = cv2.VideoCapture(input_video_path)
    if not capture.isOpened():
        print(f"ERROR: Could not open input video: {input_video_path}")
        sys.exit(1)

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Loading model: {args.model} ...")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"ERROR: Failed to load model '{args.model}'.")
        print(f"Details: {e}")
        capture.release()
        sys.exit(1)

    try:
        writer = open_video_writer(output_video_path, fps, (width, height))
    except RuntimeError as e:
        print(f"ERROR: {e}")
        capture.release()
        sys.exit(1)

    print("Model loaded successfully.")
    print(f"Input video: {input_video_path}")
    print(f"Output video: {output_video_path}")
    print(f"Output CSV: {output_csv_path}")
    print("Processing frames...")

    rows = []
    trace_history = defaultdict(dict)
    active_trace_person_id = None
    processed = 0
    failed = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        frame_number = processed
        timestamp_seconds = frame_number / fps if fps else 0.0

        try:
            if args.track:
                results = model.track(
                    source=frame,
                    conf=args.conf,
                    persist=True,
                    save=False,
                    verbose=False,
                )
            else:
                results = model.predict(
                    source=frame,
                    conf=args.conf,
                    save=False,
                    verbose=False,
                )

            result = results[0]
            annotated = result.plot()

            if result.keypoints is None or len(result.keypoints) == 0:
                rows.append(
                    {
                        "frame_number": frame_number,
                        "timestamp_seconds": round(timestamp_seconds, 4),
                        "person_id": "",
                        "person_index": "",
                    }
                )
            else:
                xy = result.keypoints.xy.cpu().numpy()
                conf = result.keypoints.conf
                conf = conf.cpu().numpy() if conf is not None else None
                people_to_process = xy.shape[0]
                if args.max_people is not None:
                    people_to_process = min(people_to_process, args.max_people)

                for person_idx in range(people_to_process):
                    person_id = get_person_id(result, person_idx)
                    angles, vertices = compute_person_angles(
                        xy, conf, person_idx, args.kp_conf
                    )

                    row = {
                        "frame_number": frame_number,
                        "timestamp_seconds": round(timestamp_seconds, 4),
                        "person_id": person_id,
                        "person_index": person_idx,
                    }
                    for angle_name, angle in angles.items():
                        row[angle_name] = "" if angle is None else angle
                    rows.append(row)

                    draw_angle_labels(annotated, person_id, angles, vertices)
                    update_traces(trace_history, person_id, angles, args.trace_length)
                    if active_trace_person_id is None:
                        active_trace_person_id = person_id

            if not args.no_traces and active_trace_person_id is not None:
                draw_trace_panel(annotated, trace_history, active_trace_person_id)

            writer.write(annotated)

        except Exception as e:
            failed += 1
            rows.append(
                {
                    "frame_number": frame_number,
                    "timestamp_seconds": round(timestamp_seconds, 4),
                    "person_id": "",
                    "person_index": "",
                }
            )
            writer.write(frame)
            print(f"WARNING: Failed to process frame {frame_number}: {e}")

        processed += 1
        if processed % 30 == 0:
            total = frame_count if frame_count else "?"
            print(f"Processed {processed}/{total} frames...")

    capture.release()
    writer.release()
    write_csv(output_csv_path, rows)

    print("\nDone!")
    print(f"Successfully read: {processed} frames")
    if failed:
        print(f"Frames with processing warnings: {failed}")
    print(f"Annotated video saved to: {output_video_path}")
    print(f"Joint angle CSV saved to: {output_csv_path}")


if __name__ == "__main__":
    main()
