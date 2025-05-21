from cv2.typing import MatLike
import cv2
from wed.utils.bounding_box import BoundingBox

def draw_bounding_boxes(img: MatLike, boxes: list[BoundingBox], color: tuple[int, int, int], thickness: int = 2) -> None:
    """Draws bounding boxes into an image.

    Args:
        img (MatLike): The image where the bounding boxes should be drawn. Modifies the image directly.
        boxes (list[BoundingBox]): The bounding boxes to draw
        color (tuple[int, int, int]): Color to draw with, the format of the image (so usually BGR).
        thickness (int, optional): Thickness of the drawn rectangle in pixels. Defaults to 2.
    """
    img_h, img_w, _ = img.shape
    for b in boxes:
        cv2.rectangle(img, b.get_rect(img_w, img_h), color, thickness)