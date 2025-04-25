from cv2.typing import MatLike
import cv2
from wed.bounding_box import BoundingBox

def draw_bounding_boxes(img: MatLike, boxes: list[BoundingBox], color: tuple[int, int, int], thickness: int = 2) -> None:
    img_h, img_w, _ = img.shape
    for b in boxes:
        cv2.rectangle(img, b.get_rect(img_w, img_h), color, thickness)