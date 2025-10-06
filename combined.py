import cv2
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import mediapipe as mp
import os
from config import CONFIG
os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # Set to -1 for CPU

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# Load YOLO and Face Detection models
model = YOLO("yolo11s.pt")  # use the nano version, change if needed
app = FaceAnalysis(name='buffalo_l')  # RetinaFace + ArcFace
app.prepare(ctx_id=0)  # GPU: 0, CPU: -1

# Constants from configuration
face_not_detected_threshold = CONFIG["face_out_of_frame"]["face_not_detected_threshold"]
face_moving_left_threshold = CONFIG["face_out_of_frame"]["face_moving_left_theshold"]
face_detection_confidence = CONFIG["face_out_of_frame"]["face_detection_confidence"]
hand_on_face_threshold = CONFIG["hand_on_face"]["hand_on_face_trigger_count"]
iou_threshold = CONFIG["hand_on_face"]["iou_threshold"]
face_moving_left_theshold = CONFIG["face_out_of_frame"]["face_moving_left_theshold"]

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


def detect_persons(model, frame, conf_threshold=0.5):
    """Run YOLOv5 detection on the frame, returns annotated frame and detections for class "person"."""
    results = model.predict(frame, conf=conf_threshold, classes=[0])  # run detection
    annotated = frame.copy()
    height, width, _ = frame.shape
    mid = width // 2

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = result.names[cls_id]
            conf = float(box.conf[0].item())

            if conf > 0.75:
                xyxy = box.xyxy[0]  # top-left & bottom-right
                x1, y1, x2, y2 = map(int, xyxy.tolist())

                # Only keep boxes whose top-left corner is in the mid-left region
                if (x1 + x2 ) / 2 < mid:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(annotated, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return annotated, results

def main(video_number):

    # Open video
    video_path = f"/data1/yasir/Data/Abnormal Behaviour3/Covering his face with the hand/Covering his face with the hand {video_number}.mp4"
    cap = cv2.VideoCapture(video_path)
    
    prevboxA, frame_count = None, 1
    values, frame_list, original_frame_list = [], [], []
    
    while True:
        ret, frame = cap.read()
        no_face_flag = True
        if not ret:
            break
        
        if frame_count % 15 == 0:
            
            original_frame_list.append(frame.copy())
            annotated_frame = frame.copy()
            boxA = None

            # Face detection
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = [f for f in app.get(image_rgb) if f.det_score > face_detection_confidence]

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
            
        frame_count+=1
    cap.release()
    
    # Save results
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    if len(values) >= hand_on_face_threshold:
        output_folder_path = os.path.join("/data1/yasir/Data/Safety Hazard/Hand on Face", video_name)

        os.makedirs(output_folder_path, exist_ok=True)
        for i, f in enumerate(frame_list):
            cv2.imwrite(os.path.join(output_folder_path, f"frame_{i}.jpg"), f)
    
    else:
        # create an empty list
        frame_list = []
        face_not_detected_count= 0
        face_moving_left_count = 0
        prev_box = None
        face_moving_left_flag = False
        continueflag = True
        for frame in original_frame_list:
            
            no_face_flag = True
            # Convert to RGB (face_recognition uses RGB)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if image_rgb is None:
                continue
            
            # annotated_frame, detections = detect_persons(model,frame, conf_threshold=0.5)

            faces = app.get(image_rgb)
            if len(faces) != 0:
                for face in faces:
                    if float(face.det_score) > face_detection_confidence: 
                        # Draw result
                        box = face.bbox.astype(int)
                        if box[0]<frame.shape[1]//2: # only consider faces in left half
                            no_face_flag = False
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                            cv2.putText(frame, f"Face {face.det_score:0.3f}", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                            if prev_box is not None:
                                if prev_box[2] > box[2]:
                                    face_moving_left_count += 1
                                else:
                                    face_moving_left_count = 0
                            prev_box = box
                            
            if no_face_flag:
                # print("No face detected", face_moving_left_count)
                face_not_detected_count+=1
                if face_moving_left_count >= face_moving_left_theshold:
                    face_moving_left_flag = True
            # cv2.imwrite(f"./{output_folder}/frame_{frame_count}.png", annotated_frame) #for my testing or debugging          
            frame_list.append(frame.copy())
        
        # Decide output folder
        print(f"Video Number: {video_number}  Face not detected count: {face_not_detected_count}, Face moving left count: {face_moving_left_flag}")
        if face_not_detected_count >= 2 and face_moving_left_flag:
            continueflag = False
            out_root = "/data1/yasir/Data/Safety Hazard/Face out of frame"
            
            output_folder_path = os.path.join(out_root, video_name)
            os.makedirs(output_folder_path, exist_ok=True)

            for i, f in enumerate(frame_list):
                filename = os.path.join(output_folder_path, f"frame_{i}.jpg")
                cv2.imwrite(filename, f)
        else:
            out_root = "/data1/yasir/Data/False Positive/"

            output_folder_path = os.path.join(out_root, os.path.splitext(os.path.basename(video_path))[0])
            os.makedirs(output_folder_path, exist_ok=True)

            for i, f in enumerate(frame_list):
                filename = os.path.join(output_folder_path, f"frame_{i}.jpg")
                cv2.imwrite(filename, f)

if __name__ == "__main__":
    for video_number in range(1, 15):
        main(video_number)
