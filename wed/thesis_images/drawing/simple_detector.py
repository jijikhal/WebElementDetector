from wed.bounding_box import BoundingBox, BoundingBoxType, RectI
from cv2.typing import MatLike
import cv2
import numpy as np
from random import randint

def draw_rect(bb: RectI, image: MatLike) -> None:
    x, y, w, h = bb
    cv2.rectangle(image, (x, y), (x+w-1, y+h-1), (255,), 1)

def find_bounding_boxes(img: MatLike) -> list[BoundingBox]:
    copy_for_show = img.copy()

    cv2.imwrite("thesis_images/images_simple/0_original.jpg", cv2.resize(img, (720, 450), interpolation=cv2.INTER_AREA))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("thesis_images/images_simple/1_gray.jpg", cv2.resize(gray, (720, 450), interpolation=cv2.INTER_AREA))
    smoothed = cv2.bilateralFilter(gray, 21, 50, 0)
    cv2.imwrite("thesis_images/images_simple/2_smoothed.jpg", cv2.resize(smoothed, (720, 450), interpolation=cv2.INTER_AREA))
    binary = cv2.Canny(smoothed, 22, 22, L2gradient=True)
    cv2.imwrite("thesis_images/images_simple/3_canny.jpg", cv2.resize(binary, (720, 450), interpolation=cv2.INTER_AREA))
    big_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    cv2.imwrite("thesis_images/images_simple/4_closing.jpg", cv2.resize(big_close, (720, 450), interpolation=cv2.INTER_AREA))
    big_count, labels, big_stats, _ = cv2.connectedComponentsWithStats(big_close, connectivity=8)
    img_h, img_w, _ = img.shape

    components = np.zeros_like(img)
    for i in range(1, big_count):
        components[labels == i] = (randint(50, 255), randint(50, 255), randint(50, 255))
    cv2.imwrite("thesis_images/images_simple/5_components.jpg", cv2.resize(components, (720, 450), interpolation=cv2.INTER_AREA))

    components = np.zeros_like(img)
    for i in range(1, big_count):
        x, y, w, h, area = big_stats[i]
        if (min(w,h) < 10 or max(w, h) < 20):
            components[labels == i] = (0, 0,255)
        else:
            components[labels == i] = (255, 255, 255)
    cv2.imwrite("thesis_images/images_simple/6_components_filtered.jpg", cv2.resize(components, (720, 450), interpolation=cv2.INTER_AREA))

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

    rects = np.zeros_like(img)

    for b in merged:
        x1, y1, x2, y2 = b.get_bb_corners()
        cv2.rectangle(rects, (round(x1*img_w), round(y1*img_h)), (round(x2*img_w), round(y2*img_h)), (255, 255, 255), 2)

    cv2.imwrite("thesis_images/images_simple/7_rects.jpg", cv2.resize(rects, (720, 450), interpolation=cv2.INTER_AREA))

    merged.sort(key=lambda x: x.area(), reverse=True)

    cv2.imshow("img", copy_for_show)
    cv2.waitKey(0)
    
    return merged

find_bounding_boxes(cv2.imread(r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\wed\cv\testing_data\is.jpg"))