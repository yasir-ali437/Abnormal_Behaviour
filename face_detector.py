from ultralytics import YOLO  # Make sure you have ultralytics installed (pip install ultralytics)

# Load YOLOv11 (pretrained on COCO)
yolo_model = YOLO("/home/yasir/retina/Abnormal_Behaviour/yolov11s-face.pt")  # you can use yolov11s.pt or bigger versions

# Run predictions on test set
results = yolo_model.predict(
    source="/data1/yasir/Data/Safety Hazard/Face out of frame/5360007/frame_10.jpg",
    save=True,
    imgsz=640,
    conf=0.25
)