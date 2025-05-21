# This file contains a script for calculating the metrics of a YOLO model
from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO(r"yolo\runs\detect\train5\weights\best.pt")
    model.info()
    model.val(data=r"yolo\dataset\dataset.yaml")