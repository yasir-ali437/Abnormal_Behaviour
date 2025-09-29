import cv2
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import mediapipe as mp
import os

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
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

def detect_persons(model, frame, conf_threshold=0.5):
    """
    Runs YOLO11 detection on the frame, returns annotated frame and detections for class "person".
    """
    results = model.predict(frame, conf=conf_threshold, classes=[0])  # run detection
    annotated = frame.copy()

    # Get image dimensions
    height, width, _ = frame.shape
    mid = width // 2

    print("Result: ", len(results))
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = result.names[cls_id]
            conf = float(box.conf[0].item())

            if conf > 0.75:
                xyxy = box.xyxy[0]  # top-left & bottom-right
                x1, y1, x2, y2 = map(int, xyxy.tolist())

                # ✅ Only keep boxes whose top-left corner is in mid-left region
                if x1 < mid:
                    # draw rectangle
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # put label + confidence
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(annotated, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return annotated, results

def main(video_number):
    main_output_dir = "./Safety Hazard"
    if not os.path.isdir(main_output_dir):
        os.makedirs(main_output_dir)
    
    # Load YOLO11 model (you can pick another variant: tiny, small, etc.)
    model = YOLO("yolo11s.pt")  # use the nano version, change if needed
    app = FaceAnalysis(name='buffalo_l')  # RetinaFace + ArcFace
    app.prepare(ctx_id=0)  # GPU: 0, CPU: -1
    # Open a video source (0 for webcam) or set path to video file / image
    video_path = f"/data1/yasir/Data/Abnormal Behaviour3/Covering his face with the hand/Covering his face with the hand {video_number}.mp4"
    cap = cv2.VideoCapture(video_path)  # change to path e.g. "video.mp4" or image
    prevboxA = None
    # create an empty list
    values = []
    frame_list = []
    frame_count = 1
    hand_on_face_threshold = 3  # Adjust this threshold based on your needs
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 15 == 0:
            # Convert to RGB (face_recognition uses RGB)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if image_rgb is None:
                continue
            
            # annotated_frame, detections = detect_persons(model,frame, conf_threshold=0.5)

            annotated_frame = frame.copy()
            boxA = None
            boxB = None

            faces = app.get(image_rgb)
            if len(faces) != 0:
                for face in faces:
                    if float(face.det_score)>0.55: 
                        # Draw result
                        box = face.bbox.astype(int)
                        if box[0]<frame.shape[1]//2: # only consider faces in left half
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

                    if x_min < width // 2:  # only consider hands in left half
                        # Draw bounding box
                        prev_x_min = x_min
                        prev_y_min = y_min
                        cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
                        cv2.putText(annotated_frame, f"Hand", (x_min, y_min -10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        boxB = (x_min, y_min, x_max, y_max)

            if boxB!=None and prevboxA!=None: #boxA==None and
                iou = intersection_over_union(prevboxA, boxB)
                if iou > 0.15:
                    values.append(iou)
                if iou > 0:
                    cv2.putText(annotated_frame, f"Alert: {iou:0.4f}", (prev_x_min, prev_y_min +10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
             
            cv2.imwrite(f"./images/frame_{frame_count}.png", annotated_frame) #for my testing or debugging          
            frame_list.append(annotated_frame.copy())
            frame_count+=1
        
        else:
            frame_count+=1
            continue

    cap.release()
    # cv2.destroyAllWindows()
    if len(values) >= hand_on_face_threshold:
        output_folder_path = os.path.join(main_output_dir, "Hand on Face",video_path.split('/')[-1].split('.')[0])
        if not os.path.isdir(output_folder_path):
            os.makedirs(output_folder_path)
        for i, f in enumerate(frame_list):
            filename = os.path.join(output_folder_path, f"frame_{i}.jpg")
            cv2.imwrite(filename, f)
    
    else:
        output_folder_path = os.path.join("./False Positive",video_path.split('/')[-1].split('.')[0])
        if not os.path.isdir(output_folder_path):
            os.makedirs(output_folder_path)
        for i, f in enumerate(frame_list):
            filename = os.path.join(output_folder_path, f"frame_{i}.jpg")
            cv2.imwrite(filename, f)
            
    print("Max value:", len(values))

if __name__ == "__main__":
    # for video_number in range(1, 15):
    #     
    main(7)
    
    
