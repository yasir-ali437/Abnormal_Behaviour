import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO-Pose model (make sure it's a *pose* model)
model = YOLO("yolo11s-pose.pt")  # you can use yolov9n-pose.pt as well

# Set video source
# video_path = "/data1/yasir/Data/Abnormal Alerts/2025-10-07/5352576/5352576_20251007_125411_video.mp4"  # <-- change to your video file or use 0 for webcam
# video_path = "/data1/yasir/Data/Abnormal Alerts/2025-10-07/5344624/5344624_20251007_075940_video.mp4"  # <-- change to your video file or use 0 for webcam
video_path = "/data1/yasir/Data/Abnormal Alerts/2025-10-07/5354195/5354195_20251007_135055_video.mp4"  # <-- change to your video file or use 0 for webcam
cap = cv2.VideoCapture(video_path)

def compute_body_angle(keypoints):
    """
    Compute the vertical rotation angle of a person from YOLO pose keypoints.
    Uses shoulder and hip midpoints to estimate torso tilt.
    """
    try:
        # Extract keypoints
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]

        # Check confidence of keypoints (score in index 2)
        if (left_shoulder[2] < 0.5 or right_shoulder[2] < 0.5 or
            left_hip[2] < 0.5 or right_hip[2] < 0.5):
            return None  # skip low-confidence detections

        # Midpoints
        shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) / 2,
                        (left_shoulder[1] + right_shoulder[1]) / 2)
        hip_mid = ((left_hip[0] + right_hip[0]) / 2,
                   (left_hip[1] + right_hip[1]) / 2)

        # Vector from shoulder to hip
        dx = hip_mid[0] - shoulder_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]

        # Angle from vertical (in degrees)
        angle = np.degrees(np.arctan2(dx, dy))  # dx/dy gives rotation from vertical
        return angle
    except Exception:
        return None
import os 
out_path = "./align"
os.makedirs(out_path, exist_ok=True)
i = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, verbose=False)

    # Iterate over all detections in frame
    for r in results:
        if r.keypoints is None:
            continue

        for kp, box in zip(r.keypoints.data.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = box.astype(int)
            # angle = compute_body_angle(kp)

            # if angle is None:
            #     continue

            # # Classify orientation
            # if abs(angle) < 15:
            #     status = "Vertical"
            #     color = (0, 255, 0)
            # elif abs(angle) < 45:
            #     status = "Tilted"
            #     color = (0, 255, 255)
            # else:
            #     status = "Horizontal"
            #     color = (0, 0, 255)

            # # Draw bounding box
            # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # cv2.putText(frame, f"{status} ({angle:.1f} deg)",
            #             (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.6, color, 2)

            # # Optional: draw torso line
            # left_shoulder = (int(kp[5][0]), int(kp[5][1]))
            # right_shoulder = (int(kp[6][0]), int(kp[6][1]))
            # left_hip = (int(kp[11][0]), int(kp[11][1]))
            # right_hip = (int(kp[12][0]), int(kp[12][1]))
            # shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) // 2,
            #                 (left_shoulder[1] + right_shoulder[1]) // 2)
            # hip_mid = ((left_hip[0] + right_hip[0]) // 2,
            #            (left_hip[1] + right_hip[1]) // 2)
            # # cv2.line(frame, shoulder_mid, hip_mid, color, 2)
            # # Draw shoulder midpoint with a dot and label
            # cv2.circle(frame, shoulder_mid, 5, (0, 255, 0), -1)  # Green dot for shoulder midpoint
            # cv2.putText(frame, 'Shoulder', (shoulder_mid[0] + 10, shoulder_mid[1] - 10), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # # Draw hip midpoint with a dot and label
            # cv2.circle(frame, hip_mid, 5, (0, 0, 255), -1)  # Red dot for hip midpoint
            # cv2.putText(frame, 'Hip', (hip_mid[0] + 10, hip_mid[1] - 10), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            # Assuming `kp` is the list or array of keypoints, where each keypoint is [x, y, confidence]

            # Define the keypoints using the format you provided
            left_shoulder = (int(kp[5][0]), int(kp[5][1]))  # Left Shoulder
            right_shoulder = (int(kp[6][0]), int(kp[6][1]))  # Right Shoulder
            left_elbow = (int(kp[7][0]), int(kp[7][1]))  # Left Elbow
            right_elbow = (int(kp[8][0]), int(kp[8][1]))  # Right Elbow
            left_wrist = (int(kp[9][0]), int(kp[9][1]))  # Left Wrist
            right_wrist = (int(kp[10][0]), int(kp[10][1]))  # Right Wrist
            left_hip = (int(kp[11][0]), int(kp[11][1]))  # Left Hip
            right_hip = (int(kp[12][0]), int(kp[12][1]))  # Right Hip
            left_knee = (int(kp[13][0]), int(kp[13][1]))  # Left Knee
            right_knee = (int(kp[14][0]), int(kp[14][1]))  # Right Knee
            left_ankle = (int(kp[15][0]), int(kp[15][1]))  # Left Ankle
            right_ankle = (int(kp[16][0]), int(kp[16][1]))  # Right Ankle
            nose = (int(kp[0][0]), int(kp[0][1]))  # Nose
            left_eye = (int(kp[1][0]), int(kp[1][1]))  # Left Eye
            right_eye = (int(kp[2][0]), int(kp[2][1]))  # Right Eye
            left_ear = (int(kp[3][0]), int(kp[3][1]))  # Left Ear
            right_ear = (int(kp[4][0]), int(kp[4][1]))  # Right Ear

            # List of keypoints and their corresponding labels
            keypoints = [
                (nose, 'Nose'),
                (left_eye, 'Left Eye'),
                (right_eye, 'Right Eye'),
                (left_ear, 'Left Ear'),
                (right_ear, 'Right Ear'),
                (left_shoulder, 'Left Shoulder'),
                (right_shoulder, 'Right Shoulder'),
                (left_elbow, 'Left Elbow'),
                (right_elbow, 'Right Elbow'),
                (left_wrist, 'Left Wrist'),
                (right_wrist, 'Right Wrist'),
                (left_hip, 'Left Hip'),
                (right_hip, 'Right Hip'),
                (left_knee, 'Left Knee'),
                (right_knee, 'Right Knee'),
                (left_ankle, 'Left Ankle'),
                (right_ankle, 'Right Ankle')
            ]

            # Draw the keypoints on the frame with dots and labels
            for keypoint, label in keypoints:
                x, y = keypoint
                if x > 0 and y > 0:  # Ensure keypoint is valid (not zero or missing)
                    # Draw a dot for each keypoint
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green dot
                    
                    # Label each keypoint with the corresponding text
                    cv2.putText(frame, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


    cv2.imwrite(f"{out_path}/Person Orientation Detection_{i}.png", frame)
    i += 1
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
# cv2.destroyAllWindows()
