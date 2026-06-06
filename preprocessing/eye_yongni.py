import cv2
import mediapipe as mp
import numpy as np
import os
import math

INPUT_VIDEO = r"/Users/neo/Desktop/video block/GRE-1-005 Anterior Visit 1.mp4"
OUTPUT_VIDEO = INPUT_VIDEO.rsplit(".", 1)[0] + "_locked_subject_blurred.mp4"

# 瘦椭圆参数
ELLIPSE_WIDTH = 280
ELLIPSE_HEIGHT = 180
Y_CENTER_OFFSET = 0.2

# 如果新检测到的脸离上一帧太远，就认为不是同一个人
MAX_JUMP_DISTANCE = 180

mp_face_detection = mp.solutions.face_detection


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def blur_ellipse(frame, cx, cy):
    h, w = frame.shape[:2]

    x1 = int(cx - ELLIPSE_WIDTH / 2)
    x2 = int(cx + ELLIPSE_WIDTH / 2)
    y1 = int(cy - ELLIPSE_HEIGHT / 2)
    y2 = int(cy + ELLIPSE_HEIGHT / 2)

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return frame

    roi = frame[y1:y2, x1:x2].copy()
    blurred = cv2.GaussianBlur(roi, (151, 151), 60)

    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    center = (roi.shape[1] // 2, roi.shape[0] // 2)
    axes = (roi.shape[1] // 2, roi.shape[0] // 2)

    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    roi[mask == 255] = blurred[mask == 255]
    frame[y1:y2, x1:x2] = roi

    return frame


def get_face_centers(results, width, height):
    centers = []

    if not results.detections:
        return centers

    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box

        x = int(bbox.xmin * width)
        y = int(bbox.ymin * height)
        bw = int(bbox.width * width)
        bh = int(bbox.height * height)

        cx = x + bw // 2
        cy = y + int(bh * Y_CENTER_OFFSET)

        centers.append((cx, cy))

    return centers


def choose_same_subject(centers, last_center):
    if not centers:
        return None

    # 第一帧：默认选画面最中间/最靠下的那个人，通常是被试
    if last_center is None:
        return min(
            centers,
            key=lambda c: abs(c[0] - FRAME_WIDTH / 2) + abs(c[1] - FRAME_HEIGHT / 2)
        )

    # 后续帧：只选离上一帧最近的脸
    nearest = min(centers, key=lambda c: distance(c, last_center))

    if distance(nearest, last_center) <= MAX_JUMP_DISTANCE:
        return nearest

    # 如果所有脸都离得太远，就不用新的脸，继续用上一帧位置
    return None


def process_video(input_path, output_path):
    global FRAME_WIDTH, FRAME_HEIGHT

    if not os.path.isfile(input_path):
        print("File not found:", input_path)
        return

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Cannot open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (FRAME_WIDTH, FRAME_HEIGHT)
    )

    print("Input:", input_path)
    print("Output:", output_path)
    print("Total frames:", total)

    last_center = None
    missing_count = 0
    max_missing_frames = 90

    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.30
    ) as face_detection:

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb)

            centers = get_face_centers(results, FRAME_WIDTH, FRAME_HEIGHT)
            current_center = choose_same_subject(centers, last_center)

            if current_center is not None:
                last_center = current_center
                missing_count = 0
            else:
                missing_count += 1
                if missing_count <= max_missing_frames:
                    current_center = last_center
                else:
                    current_center = None

            if current_center is not None:
                cx, cy = current_center
                frame = blur_ellipse(frame, cx, cy)

            out.write(frame)

            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total}")

    cap.release()
    out.release()

    print("Done!")
    print("Saved to:", output_path)


if __name__ == "__main__":
    process_video(INPUT_VIDEO, OUTPUT_VIDEO)