import cv2
import numpy as np
import os

INPUT_VIDEO = r"C:\Users\YoZhang\OneDrive - Shriners Children's\Desktop\SHAPE-UP video\clips\CAN-1-004_baseline\CAN-1-004_baseline_front_task_6.mp4"

OUTPUT_FOLDER = r"C:\Users\YoZhang\OneDrive - Shriners Children's\Desktop\SHAPE-UP video\blurred videos"

video_name = os.path.basename(INPUT_VIDEO)
OUTPUT_VIDEO = os.path.join(OUTPUT_FOLDER, video_name)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

DISPLAY_SCALE = 0.35
START_FRAME = 0

# 头被挡住/Tracker跟丢后，继续用最后位置遮挡几秒
FALLBACK_SECONDS = 3

# 马赛克大小，相对于你框住的头部区域
EYE_MASK_WIDTH_SCALE = 1.5
EYE_MASK_HEIGHT_SCALE = 1.0

# 马赛克位置，相对于你框住的头部区域
EYE_CENTER_X_RATIO = 0.50
EYE_CENTER_Y_RATIO = 0.50

# 微调：正数往右/往下，负数往左/往上，单位是像素
X_OFFSET = 0
Y_OFFSET = 0


def create_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    else:
        raise RuntimeError("Run: pip install opencv-contrib-python")


def blur_ellipse(frame, cx, cy, ellipse_width, ellipse_height):
    h, w = frame.shape[:2]

    x1 = int(cx - ellipse_width / 2)
    x2 = int(cx + ellipse_width / 2)
    y1 = int(cy - ellipse_height / 2)
    y2 = int(cy + ellipse_height / 2)

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return frame

    roi = frame[y1:y2, x1:x2].copy()

    blurred = cv2.GaussianBlur(roi, (51, 51), 30)

    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    center = (roi.shape[1] // 2, roi.shape[0] // 2)
    axes = (roi.shape[1] // 2, roi.shape[0] // 2)

    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    roi[mask == 255] = blurred[mask == 255]
    frame[y1:y2, x1:x2] = roi

    return frame


def apply_eye_mask_from_head_box(frame, bbox, fixed_width, fixed_height):
    x, y, w_box, h_box = bbox

    cx = int(x + w_box * EYE_CENTER_X_RATIO + X_OFFSET)
    cy = int(y + h_box * EYE_CENTER_Y_RATIO + Y_OFFSET)

    return blur_ellipse(
        frame,
        cx,
        cy,
        fixed_width,
        fixed_height
    )


def select_bbox(frame, window_name):
    display_frame = cv2.resize(
        frame,
        None,
        fx=DISPLAY_SCALE,
        fy=DISPLAY_SCALE
    )

    bbox_small = cv2.selectROI(
        window_name,
        display_frame,
        fromCenter=False,
        showCrosshair=True
    )

    cv2.destroyWindow(window_name)

    bbox = (
        int(bbox_small[0] / DISPLAY_SCALE),
        int(bbox_small[1] / DISPLAY_SCALE),
        int(bbox_small[2] / DISPLAY_SCALE),
        int(bbox_small[3] / DISPLAY_SCALE)
    )

    return bbox


def process_video(input_path, output_path):
    if not os.path.isfile(input_path):
        print("File not found:")
        print(input_path)
        return

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Cannot open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    max_lost_frames = int(FALLBACK_SECONDS * fps)

    print("Input:", input_path)
    print("Output:", output_path)
    print("Total frames:", total)
    print("FPS:", fps)
    print("Fallback seconds:", FALLBACK_SECONDS)
    print("Max lost frames:", max_lost_frames)

    if START_FRAME > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    ret, first_frame = cap.read()

    if not ret:
        print("Cannot read selected frame.")
        cap.release()
        return

    print("框住孩子的整个头部，不要只框眼睛。框好后按 Enter 或 Space。")

    bbox = select_bbox(
        first_frame,
        "Select WHOLE head region"
    )

    if bbox == (0, 0, 0, 0):
        print("No region selected.")
        cap.release()
        return

    fixed_mask_width = int(bbox[2] * EYE_MASK_WIDTH_SCALE)
    fixed_mask_height = int(bbox[3] * EYE_MASK_HEIGHT_SCALE)

    tracker = create_tracker()
    tracker.init(first_frame, bbox)

    last_bbox = bbox
    lost_frames = 0

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not out.isOpened():
        print("Cannot create output video.")
        cap.release()
        return

    frame_count = START_FRAME

    first_frame = apply_eye_mask_from_head_box(
        first_frame,
        bbox,
        fixed_mask_width,
        fixed_mask_height
    )

    out.write(first_frame)
    frame_count += 1

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        ok, new_bbox = tracker.update(frame)

        if ok:
            x, y, w_box, h_box = new_bbox

            if w_box > 10 and h_box > 10:
                bbox = new_bbox
                last_bbox = bbox
                lost_frames = 0
            else:
                lost_frames += 1
                bbox = last_bbox

        else:
            lost_frames += 1
            bbox = last_bbox

        if lost_frames > 0:
            print(f"Tracker lost for {lost_frames / fps:.1f} seconds")

        if lost_frames > max_lost_frames:
            print("Tracker lost too long. Please re-select head.")

            bbox_new = select_bbox(
                frame,
                "Tracker Lost - Select Head"
            )

            if bbox_new != (0, 0, 0, 0):
                bbox = bbox_new
                last_bbox = bbox

                fixed_mask_width = int(bbox[2] * EYE_MASK_WIDTH_SCALE)
                fixed_mask_height = int(bbox[3] * EYE_MASK_HEIGHT_SCALE)

                tracker = create_tracker()
                tracker.init(frame, bbox)

                lost_frames = 0
            else:
                bbox = last_bbox

        if bbox is not None:
            frame = apply_eye_mask_from_head_box(
                frame,
                bbox,
                fixed_mask_width,
                fixed_mask_height
            )

        out.write(frame)

        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Done!")
    print("Saved to:")
    print(output_path)


if __name__ == "__main__":
    process_video(INPUT_VIDEO, OUTPUT_VIDEO)