import cv2
import numpy as np
import os

# ==========================
# 输入输出路径
# ==========================

INPUT_VIDEO = r"/Users/neo/Desktop/video block/string 3 beads.mp4"

OUTPUT_FOLDER = r"/Users/neo/Desktop/video block/folder"

video_name = os.path.basename(INPUT_VIDEO)
OUTPUT_VIDEO = os.path.join(OUTPUT_FOLDER, video_name)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================
# 设置
# ==========================

# 椭圆大小 = 你手动框选区域的比例
# 0.90 = 椭圆是框选区域的90%
ELLIPSE_SCALE_WIDTH = 0.90
ELLIPSE_SCALE_HEIGHT = 0.90

# 椭圆位置微调
# 正数往右/往下，负数往左/往上
X_OFFSET = 0
Y_OFFSET = 0

# 如果你不想从第一帧开始框，可以改这里
# 例如 150 = 跳到第150帧再让你框
START_FRAME = 0


def create_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    return cv2.legacy.TrackerCSRT_create()


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
    blurred = cv2.GaussianBlur(roi, (151, 151), 60)

    mask = np.zeros(roi.shape[:2], dtype=np.uint8)

    center = (
        roi.shape[1] // 2,
        roi.shape[0] // 2
    )

    axes = (
        roi.shape[1] // 2,
        roi.shape[0] // 2
    )

    cv2.ellipse(
        mask,
        center,
        axes,
        0,
        0,
        360,
        255,
        -1
    )

    roi[mask == 255] = blurred[mask == 255]
    frame[y1:y2, x1:x2] = roi

    return frame


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

    print("Input:", input_path)
    print("Output:", output_path)
    print("Total frames:", total)
    print("FPS:", fps)
    print("Size:", width, "x", height)

    # 跳到指定帧
    if START_FRAME > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    ret, first_frame = cap.read()

    if not ret:
        print("Cannot read selected frame.")
        cap.release()
        return

    print("用鼠标框住孩子的头部/上半脸。")
    print("框好后按 Enter 或 Space。")
    print("如果框错了，按 C 取消，然后重新运行。")

    bbox = cv2.selectROI(
        "Select head region, then press Enter or Space",
        first_frame,
        fromCenter=False,
        showCrosshair=True
    )

    cv2.destroyWindow(
        "Select head region, then press Enter or Space"
    )

    if bbox == (0, 0, 0, 0):
        print("No region selected.")
        cap.release()
        return

    tracker = create_tracker()
    tracker.init(first_frame, bbox)

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_count = START_FRAME

    # 如果 START_FRAME > 0，前面的帧不输出。
    # 也就是说，输出视频从你框选的那一帧开始。

    x, y, w_box, h_box = bbox

    cx = int(x + w_box / 2 + X_OFFSET)
    cy = int(y + h_box / 2 + Y_OFFSET)

    ellipse_width = int(w_box * ELLIPSE_SCALE_WIDTH)
    ellipse_height = int(h_box * ELLIPSE_SCALE_HEIGHT)

    first_frame = blur_ellipse(
        first_frame,
        cx,
        cy,
        ellipse_width,
        ellipse_height
    )

    out.write(first_frame)
    frame_count += 1

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        ok, bbox = tracker.update(frame)

        if ok:
            x, y, w_box, h_box = bbox

            cx = int(x + w_box / 2 + X_OFFSET)
            cy = int(y + h_box / 2 + Y_OFFSET)

            ellipse_width = int(w_box * ELLIPSE_SCALE_WIDTH)
            ellipse_height = int(h_box * ELLIPSE_SCALE_HEIGHT)

            frame = blur_ellipse(
                frame,
                cx,
                cy,
                ellipse_width,
                ellipse_height
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