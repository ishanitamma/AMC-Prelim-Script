import cv2
import mediapipe as mp
import numpy as np
import sys

def blur_face(frame, box):
    x1,y1,x2,y2 = box
    if x2 <= x1 or y2 <= y1:
        return
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(region, (151,151), 50)

def smooth_box(prev, curr, alpha=0.6):
    return tuple(int(alpha*c + (1-alpha)*p) for p,c in zip(prev,curr))

input_video = sys.argv[1]
output_video = sys.argv[2]

cap = cv2.VideoCapture(input_video)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))

last_box = None

with mp.solutions.face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.45
) as detector:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        if results.detections:
            det = results.detections[0]
            bbox = det.location_data.relative_bounding_box

            face_w = int(bbox.width * w)
            face_h = int(bbox.height * h)

            pad_x = int(face_w * 0.35)
            pad_y = int(face_h * 0.40)

            box = (
                max(0, int(bbox.xmin*w)-pad_x),
                max(0, int(bbox.ymin*h)-pad_y),
                min(w, int((bbox.xmin+bbox.width)*w)+pad_x),
                min(h, int((bbox.ymin+bbox.height)*h)+pad_y)
            )

            if last_box is not None:
                box = smooth_box(last_box, box)

            last_box = box

        if last_box is not None:
            blur_face(frame, last_box)

        out.write(frame)

cap.release()
out.release()
print("Saved:", output_video)
