import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture("/home/adlytic/Yasir Adlytic/Abnormal_Behaviour/Dataset/Abnormal Behavior/Safety Hazard/Looking Backward/2 looking backward.mp4")  # or your video path

with mp_pose.Pose(static_image_mode=False, 
                  min_detection_confidence=0.5, 
                  min_tracking_confidence=0.5) as pose:

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
         # Get image dimensions
        height, width, _ = frame.shape
        mid = width // 2

        left_half = frame[:, :mid]
        rgb = cv2.cvtColor(left_half, cv2.COLOR_BGR2RGB)
        h, w, _ = left_half.shape
        results = pose.process(rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # left shoulder = 11, right shoulder = 12
            ls, rs = lm[11], lm[12]
            neck_x = int(((ls.x + rs.x) / 2) * w)
            neck_y = int(((ls.y + rs.y) / 2) * h)

            shoulder_dist = int(abs(ls.x - rs.x) * w)
            head_h = int(1.2 * shoulder_dist)
            head_w = int(0.8 * shoulder_dist)

            x1 = max(neck_x - head_w//2, 0)
            x2 = min(neck_x + head_w//2, w-1)
            y2 = neck_y
            y1 = max(neck_y - head_h, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        cv2.imshow("Head Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
