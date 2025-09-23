from ultralytics import YOLO

# Load a pretrained YOLOv11 model (on COCO)
model = YOLO("yolo11s.pt")   # you can also try yolo11s.pt, yolo11m.pt for bigger models

# Train on your bottle dataset
model.train(
    data="bottle.yaml",   # path to dataset YAML
    epochs=50,            # increase if needed
    imgsz=640,            # input image size
    batch=16,             # adjust based on GPU memory
    name="bottle_yolo11"  # run name
)
model.save("yolo11s_bottle.pt")
