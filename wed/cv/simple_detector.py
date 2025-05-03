# Detector used for the environment with rectangles based on real websites (Section 4.16)
from wed.bounding_box import BoundingBox, BoundingBoxType, RectI
from cv2.typing import MatLike
import cv2

def draw_rect(bb: RectI, image: MatLike) -> None:
    """Draws a rectangle in the image

    Args:
        bb (RectI): Rectangle in a (x, y, w, h) format in pixels
        image (MatLike): Image to draw the rectangle into
    """
    x, y, w, h = bb
    cv2.rectangle(image, (x, y), (x+w-1, y+h-1), (255,), 1)

def find_bounding_boxes(img: MatLike) -> list[BoundingBox]:
    """Finds elements in an website screenshot image. Does not work that well.

    Args:
        img (MatLike): Image to analyze

    Returns:
        list[BoundingBox]: List of the found element bounding boxes
    """
    copy_for_show = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.bilateralFilter(gray, 21, 50, 0)
    binary = cv2.Canny(smoothed, 22, 22, L2gradient=True)
    big_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    big_count, _, big_stats, _ = cv2.connectedComponentsWithStats(big_close, connectivity=8)
    img_h, img_w, _ = img.shape

    result: list[BoundingBox] = [BoundingBox((0, 0, 1, 1), BoundingBoxType.TWO_CORNERS)]

    for i in range(1, big_count):
        x, y, w, h, area = big_stats[i]
        if (min(w,h) < 10 or max(w, h) < 20):
            cv2.rectangle(copy_for_show, (x, y), (x + w + 1, y + h + 1), (0, 0, 255), 1)
            continue
        result.append(BoundingBox((x/img_w, y/img_h, (w+1)/img_w, (h+1)/img_h), BoundingBoxType.TOP_LEFT))
        cv2.rectangle(copy_for_show, (x, y), (x + w + 1, y + h + 1), (0, 255, 0), 2)

    merged: list[BoundingBox] = []
    used = set()
    result.sort(key=lambda x: x.area())
    for b in result:
        if (b in used):
            continue
        used.add(b)
        to_merge_with = [x for x in result if b.is_intersecting(x) and x not in used]
        m = b
        for t in to_merge_with:
            used.add(t)
            m = b.merge(t)
        merged.append(m)

    for b in merged:
        x1, y1, x2, y2 = b.get_bb_corners()
        cv2.rectangle(copy_for_show, (round(x1*img_w), round(y1*img_h)), (round(x2*img_w), round(y2*img_h)), (255, 0, 0), 2)

    merged.sort(key=lambda x: x.area(), reverse=True)

    #cv2.imshow("img", copy_for_show)
    #cv2.waitKey(0)
    
    return merged