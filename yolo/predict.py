# https://docs.ultralytics.com/models/yolov8/#performance-metrics

from ultralytics import YOLO
from ultralytics.engine.results import Results
from ReinforcementLearning.bounding_box import BoundingBox, BoundingBoxType
import cv2

if __name__ == "__main__":

    model = YOLO(r"runs\detect\train5\weights\best.pt")

    img_path = r"dataset\images\train\3live.ru.jpg"
    img = cv2.imread(img_path)
    results: list[Results] = model(img_path)
    img_h, img_w, _ = img.shape
    for b in results[0].boxes.xywhn:
        bb = BoundingBox(b.tolist(), BoundingBoxType.CENTER)
        cv2.rectangle(img, bb.get_rect(img_w, img_h), (255, 255, 0), 2)

    cv2.imshow("lol", img)
    cv2.waitKey(0)

