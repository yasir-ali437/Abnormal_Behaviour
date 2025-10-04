import cv2
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import os

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
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return annotated, results

model = YOLO("yolo11s.pt")  # use the nano version, change if needed
app = FaceAnalysis(name='buffalo_l')  # RetinaFace + ArcFace
app.prepare(ctx_id=0)  # GPU: 0, CPU: -1
    
def main(video_number):
    
    # Load YOLO11 model (you can pick another variant: tiny, small, etc.)
    # Open a video source (0 for webcam) or set path to video file / image
    video_path = f"/data1/yasir/Data/Abnormal Behaviour3/Face out of frame/Face out of frame {video_number}.mp4"
    cap = cv2.VideoCapture(video_path)  # change to path e.g. "video.mp4" or image

    # create an empty list
    values = []
    frame_list = []
    frame_count = 1
    face_not_detected_count= 0
    face_moving_left_count = 0
    prev_box = None
    face_moving_left_flag = False
    
    while True:
        ret, frame = cap.read()
        no_face_flag = True
        if not ret:
            break
        
        if frame_count % 15 == 0:
            # Convert to RGB (face_recognition uses RGB)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if image_rgb is None:
                continue
            
            # annotated_frame, detections = detect_persons(model,frame, conf_threshold=0.5)

            faces = app.get(image_rgb)
            if len(faces) != 0:
                for face in faces:
                    if float(face.det_score)>0.40: 
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
                print("No face detected", face_moving_left_count)
                face_not_detected_count+=1
                if face_moving_left_count>=1:
                    face_moving_left_flag = True
            # cv2.imwrite(f"./{output_folder}/frame_{frame_count}.png", annotated_frame) #for my testing or debugging          
            frame_list.append(frame.copy())
            frame_count+=1
        
        else:
            frame_count+=1
            continue
        
        
    cap.release()
    
    # Decide output folder
    print(f"Video Number: {video_number}  Face not detected count: {face_not_detected_count}, Face moving left count: {face_moving_left_flag}")
    if face_not_detected_count >= 2 and face_moving_left_flag:
        out_root = "/data1/yasir/Data/Safety Hazard/Face out of frame"
    else:
        out_root = "/data1/yasir/Data/False Positive/"

    output_folder_path = os.path.join(out_root, os.path.splitext(os.path.basename(video_path))[0])
    os.makedirs(output_folder_path, exist_ok=True)

    for i, f in enumerate(frame_list):
        filename = os.path.join(output_folder_path, f"frame_{i}.jpg")
        cv2.imwrite(filename, f)
            

if __name__ == "__main__":
    for video_number in range(1, 22):
        main(video_number)
    
    
