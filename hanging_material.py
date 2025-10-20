import cv2
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import mediapipe as mp
import os
os.environ["CuDA_VISIBLE_DEVICES"] = "4"
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Load YOLO11 model (you can pick another variant: tiny, small, etc.)
model = YOLO("/home/yasir/retina/hanging_material_training/runs/detect/hanging_material_yolo11/weights/best.pt")  # use the nano version, change if needed
app = FaceAnalysis(name='buffalo_l')  # RetinaFace + ArcFace
app.prepare(ctx_id=0)  # GPU: 0, CPU: -1
    
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=2,
#     min_detection_confidence=0.3,
#     min_tracking_confidence=0.3
# )

bottle_on_face_threshold = 3
bottle_conf_threshold = 0.5

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

def detect(model, frame):
    """
    Runs YOLO11 detection on the frame, returns annotated frame and detections for class "bottle".
    """
    results = model.predict(frame, conf=bottle_conf_threshold, classes=[0])  # run detection
    annotated = frame.copy()

    # Get image dimensions
    height, width, _ = frame.shape
    mid = width // 2
    detection = False
    
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = result.names[cls_id]
            conf = float(box.conf[0].item())

            # if conf > 0.5:
            xyxy = box.xyxy[0]  # top-left & bottom-right
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            detection = True
            # ✅ Only keep boxes whose top-left corner is in mid-left region
            # if x1 < mid:
            # draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # put label + confidence
            label = f"{class_name} {conf:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return annotated, detection

def main(video_number):    

    # Open a video source (0 for webcam) or set path to video file / image
    video_path = f"/data1/yasir/Data/Abnormal Behavior1/Abnormal Behavior/Safety Hazard/Hanging Material/{video_number} hanging material.mp4"
    cap = cv2.VideoCapture(video_path)  # change to path e.g. "video.mp4" or image
    prevboxA = None
    # create an empty list
    values = []
    frame_list = []
    frame_count = 1
    hanging_material_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 15 == 0:
            
            annotated_frame, detection = detect(model, frame)
            if detection:
                hanging_material_count += 1
                
            frame_list.append(annotated_frame)
            # Convert to RGB (face_recognition uses RGB)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if image_rgb is None:
                continue

            faces = app.get(image_rgb)
            if len(faces) != 0:
                for face in faces:
                    if float(face.det_score)>0.55: 
                        # Draw result
                        box = face.bbox.astype(int)
                        if box[0]<frame.shape[1]//2: # only consider faces in left half
                            cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                            cv2.putText(annotated_frame, f"Face {face.det_score:0.3f}", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            
        frame_count+=1
        

    cap.release()
    # cv2.destroyAllWindows()
    if hanging_material_count >= 5:
        output_folder_path = os.path.join("/data1/yasir/Data/Safety Hazard/", "Hanging Material",video_path.split('/')[-1].split('.')[0])
    else:
        output_folder_path = os.path.join("/data1/yasir/Data/False Positive/",video_path.split('/')[-1].split('.')[0])
        
    if not os.path.isdir(output_folder_path):
        os.makedirs(output_folder_path)
    for i, f in enumerate(frame_list):
        filename = os.path.join(output_folder_path, f"frame_{i}.jpg")
        cv2.imwrite(filename, f)
            
    print("hanging_material_count :", hanging_material_count)

if __name__ == "__main__":
    for video_number in range(1, 33):
        main(video_number)
    
    
