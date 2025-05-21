# This script can be used to test a trained YOLO model.

from ultralytics import YOLO
from ultralytics.engine.results import Results
from wed.utils.bounding_box import BoundingBox, BoundingBoxType
from wed.utils.draw_bb import draw_bounding_boxes
import cv2
from os.path import join, isfile
from os import listdir

if __name__ == "__main__":

    model = YOLO(r"yolo\runs\detect\train5\weights\best.pt")

    images = r"yolo\dataset\images\test"
    paths = [join(images, f) for f in listdir(
        images) if isfile(join(images, f))]
    for i in paths:
        img_path = i
        img = cv2.imread(img_path)
        results: list[Results] = model(img_path, verbose=False)
        img_h, img_w, _ = img.shape
        bbs: list[BoundingBox] = [BoundingBox(b.tolist(), BoundingBoxType.CENTER) for b in results[0].boxes.xywhn]

        draw_bounding_boxes(img, bbs, (255, 255, 0), 2)
        

        cv2.imshow("lol", img)
        if cv2.waitKey(0) == ord('q'):
            break

    cv2.destroyAllWindows()

