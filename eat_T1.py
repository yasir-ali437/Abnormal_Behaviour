import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO  # Make sure you have ultralytics installed (pip install ultralytics)

# Load YOLOv11 (pretrained on COCO)
yolo_model = YOLO("yolo11s.pt")  # you can use yolov11s.pt or bigger versions

# Initialize mediapipe holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture("/home/adlytic/Yasir Adlytic/Abnormal_Behaviour/Dataset/Abnormal Behavior/Safety Hazard/Eat & Drink/3 eat & drink.mp4")  # or replace with 'video.mp4'

def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
    i=0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # cv2.imwrite(f'./images/frame{i}.png',frame)
        i+=1
        # Get image dimensions
        # height, width, _ = frame.shape
        # mid = width // 2

        # frame = frame[:, :mid]

        # YOLO detection (bottle check)
        results_yolo = yolo_model(frame, classes=[39],verbose=False,conf=0.25)
        bottle_detected = False

        for result in results_yolo:
            for box in result.boxes:
                cls_id = int(box.cls[0])  # class id
                label = result.names[cls_id]  # class name
                # if label == "bottle":  # check if it's a bottle
                bottle_detected = True
                # Draw bounding box
                conf = box.conf[0].item()
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"Bottle: {conf:0.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Mediapipe holistic
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        eating_detected = False

        if results.face_landmarks and (results.left_hand_landmarks or results.right_hand_landmarks):
            # Mouth center (upper + lower lip average)
            mouth_top = results.face_landmarks.landmark[13]  # Upper lip
            mouth_bottom = results.face_landmarks.landmark[14]  # Lower lip
            mouth_center = ((mouth_top.x + mouth_bottom.x) / 2,
                            (mouth_top.y + mouth_bottom.y) / 2)

            h, w, _ = image.shape
            mouth_px = (int(mouth_center[0] * w), int(mouth_center[1] * h))

            # Check left and right hands
            for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
                if hand_landmarks:
                    fingertip = hand_landmarks.landmark[8]  # index fingertip
                    finger_px = (int(fingertip.x * w), int(fingertip.y * h))

                    # Distance between fingertip and mouth
                    dist = euclidean_distance(mouth_px, finger_px)

                    if dist < 50 and bottle_detected:  # require both gesture + bottle
                        eating_detected = True
                        cv2.putText(image, "Eating Detected", (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        cv2.circle(image, mouth_px, 8, (0, 255, 0), -1)
                        cv2.circle(image, finger_px, 8, (0, 0, 255), -1)

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow("Eating Detection (YOLO + MediaPipe)", image)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
