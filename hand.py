import cv2
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import mediapipe as mp
import os
from config import CONFIG

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

hand_on_face_threshold = CONFIG["hand_on_face"]["hand_on_face_trigger_count"]
iou_threshold = CONFIG["hand_on_face"]["iou_threshold"]

# Load models
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)  # GPU: 0, CPU: -1

def intersection_over_union(boxA, boxB):
    """Compute IoU between two bounding boxes (x1,y1,x2,y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width, inter_height = max(0, xB - xA), max(0, yB - yA)
    inter_area = inter_width * inter_height

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = boxA_area + boxB_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def main(video_number):
    main_output_dir = "./Safety Hazard"
    os.makedirs(main_output_dir, exist_ok=True)

    # Open video
    video_path = f"/data1/yasir/Data/Abnormal Behaviour3/Eat & Drink/Eat & drink {video_number}.mp4"
    cap = cv2.VideoCapture(video_path)

    prevboxA, frame_count = None, 1
    values, frame_list = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 15 == 0:
            annotated_frame = frame.copy()
            boxA = None

            # Face detection
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = [f for f in app.get(image_rgb) if f.det_score > 0.55]

            for face in faces:
                box = face.bbox.astype(int)
                if box[0] < frame.shape[1] // 2:  # left half
                    cv2.rectangle(annotated_frame, (box[0], box[1]),
                                  (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Face {face.det_score:.3f}",
                                (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    boxA = tuple(box)

            # Use last face box if none detected
            if boxA:
                prevboxA = boxA

            if prevboxA:
                cxA = (prevboxA[0] + prevboxA[2]) / 2
                cyA = (prevboxA[1] + prevboxA[3]) / 2
            else:
                cxA = cyA = None

            # Hand detection (left half only)
            left_half = frame[:, :frame.shape[1] // 2]
            hand_rgb = cv2.cvtColor(left_half, cv2.COLOR_BGR2RGB)
            results = hands.process(hand_rgb)

            boxB = None
            if results.multi_hand_landmarks and cxA is not None:
                closest_hand, min_dist = None, float("inf")

                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                      results.multi_handedness):
                    x_coords = [lm.x for lm in hand_landmarks.landmark]
                    y_coords = [lm.y for lm in hand_landmarks.landmark]
                    h, w, _ = hand_rgb.shape
                    x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
                    y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

                    cxB, cyB = (x_min + x_max) / 2, (y_min + y_max) / 2
                    dist = ((cxA - cxB) ** 2 + (cyA - cyB) ** 2) ** 0.5

                    if dist < min_dist:
                        min_dist = dist
                        closest_hand = (x_min, y_min, x_max, y_max, handedness)

                if closest_hand:
                    x_min, y_min, x_max, y_max, handedness = closest_hand
                    label, score = handedness.classification[0].label, handedness.classification[0].score
                    cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Hand {score:.2f}",
                                (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    boxB = (x_min, y_min, x_max, y_max)

            # Hand-on-face IoU check
            if boxB and prevboxA:
                iou = intersection_over_union(prevboxA, boxB)
                if iou > iou_threshold:
                    values.append(iou)
                if iou > 0:
                    cv2.putText(annotated_frame, f"Alert: {iou:.4f}",
                                (x_min, y_min + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            frame_list.append(annotated_frame)
            cv2.imwrite(f"./images/frame_{frame_count}.png", annotated_frame)  # Debugging
        frame_count += 1

    cap.release()

    # Save results
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if len(values) >= hand_on_face_threshold:
        output_folder_path = os.path.join(main_output_dir, "Hand on Face", video_name)
    else:
        output_folder_path = os.path.join("./False Positive", video_name)

    os.makedirs(output_folder_path, exist_ok=True)
    for i, f in enumerate(frame_list):
        cv2.imwrite(os.path.join(output_folder_path, f"frame_{i}.jpg"), f)



if __name__ == "__main__":
    for video_number in range(1, 7):
        main(video_number)
