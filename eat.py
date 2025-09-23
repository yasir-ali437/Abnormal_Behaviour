import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture("/home/adlytic/Yasir Adlytic/Abnormal_Behaviour/Dataset/Abnormal Behavior/Safety Hazard/Eat & Drink/1 eat & drink.mp4")  # Change to "video.mp4" for a video file

def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get image dimensions
        height, width, _ = frame.shape
        mid = width // 2

        left_half = frame[:, :mid]

        # Convert the frame to RGB
        image = cv2.cvtColor(left_half, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)

        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        eating_detected = False

        if results.face_landmarks and (results.left_hand_landmarks or results.right_hand_landmarks):
            # Get mouth center (average of upper & lower lip midpoints)
            mouth_top = results.face_landmarks.landmark[13]  # Upper lip
            mouth_bottom = results.face_landmarks.landmark[14]  # Lower lip
            mouth_center = ((mouth_top.x + mouth_bottom.x) / 2,
                            (mouth_top.y + mouth_bottom.y) / 2)

            h, w, _ = image.shape
            mouth_px = (int(mouth_center[0] * w), int(mouth_center[1] * h))

            # Check left and right hands
            for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
                if hand_landmarks:
                    # Take index fingertip (landmark 8)
                    fingertip = hand_landmarks.landmark[8]
                    finger_px = (int(fingertip.x * w), int(fingertip.y * h))

                    # Compute distance
                    dist = euclidean_distance(mouth_px, finger_px)
                    print("Euclidean Distance: ",dist)
                    if dist < 50:  # Threshold in pixels (adjust!)
                        eating_detected = True
                        cv2.putText(image, "Eating Gesture Detected", (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.circle(image, mouth_px, 8, (0, 255, 0), -1)
                        cv2.circle(image, finger_px, 8, (0, 0, 255), -1)

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow('Eating Detection', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
