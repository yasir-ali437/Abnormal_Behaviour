import cv2
import mediapipe as mp


# Initialize mediapipe face detection + face mesh
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

def classify_head_orientation(image):
    """
    Classify head orientation:
    0 = front, 1 = side, 2 = back
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Face Mesh (gives detailed landmarks)
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

        result = face_mesh.process(img_rgb)

        if not result.multi_face_landmarks:
            # No face → probably BACK
            return 2, 0

        # Get landmarks (468 points per face)
        face_landmarks = result.multi_face_landmarks[0]

        # Indices for key points
        LEFT_EYE = 33
        RIGHT_EYE = 263
        NOSE_TIP = 1

        # Get coordinates
        h, w, _ = image.shape
        left_eye = face_landmarks.landmark[LEFT_EYE]
        right_eye = face_landmarks.landmark[RIGHT_EYE]
        nose_tip = face_landmarks.landmark[NOSE_TIP]

        # Convert to pixel coords
        lx, ly = int(left_eye.x * w), int(left_eye.y * h)
        rx, ry = int(right_eye.x * w), int(right_eye.y * h)
        nx, ny = int(nose_tip.x * w), int(nose_tip.y * h)

        # Rule-based classification
        eye_dist = abs(lx - rx)
        print(f"Eye distance: {eye_dist}")
        if eye_dist > w * 0.15:
            # Both eyes visible → FRONT
            return (0, eye_dist)
        elif eye_dist > w * 0.05:
            # One eye more visible → SIDE
            return (1, eye_dist)
        else:
            # Very skewed or hidden face → BACK
            return (2, eye_dist)


# -------------------------
# Test on a video
# -------------------------
cap = cv2.VideoCapture("/data1/yasir/Data/Abnormal Behaviour3/Looking Backward/Looking backward 1.mp4")
i=1
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cls, eye_dist = classify_head_orientation(frame)
    label = {0: "Front", 1: "Side", 2: "Back"}[cls]

    cv2.putText(frame, label +" "+str(eye_dist), (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(f"./images/Head_Orientation_{i}.png", frame)
    i+=1
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break

cap.release()
