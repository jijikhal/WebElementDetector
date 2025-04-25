# https://docs.ultralytics.com/models/yolov8/#performance-metrics

from ultralytics import YOLO
from ultralytics.engine.results import Results
from wed.bounding_box import BoundingBox, BoundingBoxType
from wed.utils.draw_bb import draw_bounding_boxes
import cv2

if __name__ == "__main__":

    model = YOLO(r"runs\detect\train5\weights\best.pt")

    img_path = r"dataset\images\train\3live.ru.jpg"
    img = cv2.imread(img_path)
    results: list[Results] = model(img_path)
    img_h, img_w, _ = img.shape
    bbs = [b.tolist for b in results[0].boxes.xywhn]

    draw_bounding_boxes(img, bbs, (255, 255, 0), 2)
    

    cv2.imshow("lol", img)
    cv2.waitKey(0)

