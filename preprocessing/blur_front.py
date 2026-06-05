import cv2
import mediapipe as mp
import numpy as np
import sys

LEFT_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
RIGHT_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]

def blur_eye_region(frame, pts, scale=2.0):
    pts = np.array(pts, np.int32)
    hull = cv2.convexHull(pts)
    cx = np.mean(hull[:,0,0]); cy = np.mean(hull[:,0,1])
    expanded = []
    for p in hull[:,0]:
        x,y = p
        expanded.append([int(cx + scale*(x-cx)), int(cy + scale*(y-cy))])
    expanded = np.array(expanded, np.int32)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, expanded, 255)

    blurred = cv2.GaussianBlur(frame, (151,151), 50)
    frame[mask > 0] = blurred[mask > 0]

input_video = sys.argv[1]
output_video = sys.argv[2]

cap = cv2.VideoCapture(input_video)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))

mp_face_mesh = mp.solutions.face_mesh
with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            left_pts = [[int(face.landmark[i].x*w), int(face.landmark[i].y*h)] for i in LEFT_EYE]
            right_pts = [[int(face.landmark[i].x*w), int(face.landmark[i].y*h)] for i in RIGHT_EYE]

            blur_eye_region(frame, left_pts)
            blur_eye_region(frame, right_pts)

        out.write(frame)

cap.release()
out.release()
print("Saved:", output_video)
