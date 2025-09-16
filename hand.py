import cv2
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def intersection_over_union(boxA, boxB):
    # box format: (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute intersection area
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    # Areas of both boxes
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute union area
    union_area = boxA_area + boxB_area - inter_area

    # Avoid division by zero
    if union_area == 0:
        return 0.0

    iou = inter_area / union_area
    return iou

def detect_persons(model, frame, original_frame, conf_threshold=0.5):
    """
    Runs YOLO11 detection on the frame, returns annotated frame and detections for class "person".
    """
    results = model.predict(frame, conf=conf_threshold, classes=[0])  # run detection
    annotated = original_frame.copy()

    print("Result: ",len(results))
    for result in results:
        # result.boxes.cls gives class indices, result.names maps them to names
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = result.names[cls_id]
            if float(box.conf[0].item()) >0.75: #class_name == 'person':
                xyxy = box.xyxy[0]  # top-left & bottom-right
                x1, y1, x2, y2 = map(int, xyxy.tolist())
                # draw rectangle
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
                # put label + confidence
                conf = box.conf[0].item()
                label = f"{class_name} {conf:.2f}"
                cv2.putText(annotated, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return annotated, results

def main():
    # Load YOLO11 model (you can pick another variant: tiny, small, etc.)
    model = YOLO("yolo11s.pt")  # use the nano version, change if needed
    app = FaceAnalysis(name='buffalo_l')  # RetinaFace + ArcFace
    app.prepare(ctx_id=0)  # GPU: 0, CPU: -1
    # Open a video source (0 for webcam) or set path to video file / image
    cap = cv2.VideoCapture("/home/adlytic/Yasir Adlytic/Abnormal_Behaviour/Dataset/Abnormal Behavior/Safety Hazard/Covering his face with the hand/1 covering his face with the hand.mp4")  # change to path e.g. "video.mp4" or image
    prevboxA = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get image dimensions
        height, width, _ = frame.shape
        mid = width // 2

        left_half = frame[:, :mid]

        # Convert to RGB (face_recognition uses RGB)
        image_rgb = cv2.cvtColor(left_half, cv2.COLOR_BGR2RGB)

        if image_rgb is None:
            continue
        
        annotated_frame, detections = detect_persons(model, left_half,frame, conf_threshold=0.5)

        boxA = None
        boxB = None

        faces = app.get(image_rgb)
        if len(faces) != 0:
            if float(faces[0].det_score)>0.55: 
                face = faces[0] 
                # Draw result
                box = face.bbox.astype(int)
                cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Face {face.det_score:0.3f}", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                boxA = (box[0], box[1], box[2], box[3])
        
        if boxA:
            prevboxA = boxA

        results = hands.process(image_rgb)
        height, width, _ = image_rgb.shape
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get bounding box from landmarks
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]

                x_min, x_max = int(min(x_coords) * width), int(max(x_coords) * width)
                y_min, y_max = int(min(y_coords) * height), int(max(y_coords) * height)

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
                cv2.putText(annotated_frame, f"Hand", (x_min, y_min -10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                boxB = (x_min, y_min, x_max, y_max)

        if boxA==None and boxB!=None and prevboxA!=None:
            iou = intersection_over_union(prevboxA, boxB)
            cv2.putText(annotated_frame, f"IOU: {iou:0.4f}", (x_min, y_min +10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Person Detection YOLO11", annotated_frame)
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
