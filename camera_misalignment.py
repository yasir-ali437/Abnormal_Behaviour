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
            angle = compute_body_angle(kp)

            if angle is None:
                continue

            # Classify orientation
            if abs(angle) < 15:
                status = "Vertical"
                color = (0, 255, 0)
            elif abs(angle) < 45:
                status = "Tilted"
                color = (0, 255, 255)
            else:
                status = "Horizontal"
                color = (0, 0, 255)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{status} ({angle:.1f} deg)",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

            # Optional: draw torso line
            left_shoulder = (int(kp[5][0]), int(kp[5][1]))
            right_shoulder = (int(kp[6][0]), int(kp[6][1]))
            left_hip = (int(kp[11][0]), int(kp[11][1]))
            right_hip = (int(kp[12][0]), int(kp[12][1]))
            shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) // 2,
                            (left_shoulder[1] + right_shoulder[1]) // 2)
            hip_mid = ((left_hip[0] + right_hip[0]) // 2,
                       (left_hip[1] + right_hip[1]) // 2)
            cv2.line(frame, shoulder_mid, hip_mid, color, 2)

    cv2.imwrite(f"{out_path}/Person Orientation Detection_{i}.png", frame)
    i += 1
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
# cv2.destroyAllWindows()
