# This file contains the script for training the YOLOv8 detector used in Section 5.2.
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.info()
    results = model.train(data=r"yolo\dataset\dataset.yaml", epochs=10000, imgsz=640)