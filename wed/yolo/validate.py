# This file contains a script for calculating the metrics of a YOLO model
from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("runs/detect/8000_dataset/weights/best.pt")
    model.info()
    model.val()