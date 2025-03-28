import cv2
from cv2.typing import MatLike
import numpy as np
import random
from os import listdir
from os.path import isfile, join

from bounding_box import BoundingBox, BoundingBoxType



def find_bounding_boxes(img: MatLike) -> list[BoundingBox]:
    result: list[BoundingBox] = []
    img_h, img_w, _ = img.shape

    copy_for_show = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.bilateralFilter(gray, 21, 50, 10)
    binary = cv2.Canny(smoothed, 22, 22, L2gradient=True)

    # Find elements using connected components
    """closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    count, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if (min(w,h) < 10 or max(w, h) < 20):
            cv2.rectangle(copy_for_show, (x, y), (x + w + 1, y + h + 1), (0, 0, 255), 1)
            continue
        result.append(BoundingBox((x/img_w, y/img_h, (w+1)/img_w, (h+1)/img_h), BoundingBoxType.TOP_LEFT))
        cv2.rectangle(copy_for_show, (x, y), (x + w + 1, y + h + 1), (0, 255, 0), 2)"""

    # Find using contours
    closed = cv2.morphologyEx(binary, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)))
    cv2.imshow("edges", closed)
    small_trash: list[BoundingBox] = []
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if cv2.contourArea(cv2.boxPoints(cv2.minAreaRect(c)))/(w*h) < 0.75:
            cv2.rectangle(copy_for_show, (x, y), (x + w + 1, y + h + 1), (255, 0, 255), 1)
            small_trash.append(BoundingBox((x/img_w, y/img_h, (w+1)/img_w, (h+1)/img_h), BoundingBoxType.TOP_LEFT))
            continue
        if (min(w,h) < 10 or max(w, h) < 20):
            cv2.rectangle(copy_for_show, (x, y), (x + w + 1, y + h + 1), (53, 132, 242), 1)
            small_trash.append(BoundingBox((x/img_w, y/img_h, (w+1)/img_w, (h+1)/img_h), BoundingBoxType.TOP_LEFT))
            continue
        result.append(BoundingBox((x/img_w, y/img_h, (w+1)/img_w, (h+1)/img_h), BoundingBoxType.TOP_LEFT))
        cv2.rectangle(copy_for_show, (x, y), (x + w + 1, y + h + 1), (255, 0, 0), 2)

    merged: list[BoundingBox] = [BoundingBox((0,0,1,1), BoundingBoxType.TOP_LEFT)]
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
        #cv2.rectangle(copy_for_show, (round(x1*img_w), round(y1*img_h)), (round(x2*img_w), round(y2*img_h)), (255, 255, 0), 2)
    merged.sort(key=lambda x: x.area(), reverse=True)

    trash_count: dict[BoundingBox, int] = {x:0 for x in merged}
    for t in small_trash:
        parent = min([x for x in merged if x.fully_contains(t)], key=lambda x: x.area())
        trash_count[parent] += 1

    for m in merged:
        if trash_count[m] > 10:
            x, y, w, h = m.get_bb_tl()
            cv2.rectangle(copy_for_show, m.get_rect(img_w, img_h), (0, 0, 255), 2)

    cv2.imshow("img", copy_for_show)
    
    
    return merged

if __name__ == "__main__":
    dataset_folder = r"C:\Users\halabala\Documents\GitHub\WebElementDetector\ReinforcementLearning\dataset_big"
    paths = [join(dataset_folder, f) for f in listdir(dataset_folder) if isfile(join(dataset_folder, f))]
    for p in paths:
        img = cv2.imread(p)
        boxes = find_bounding_boxes(img)
        if cv2.waitKey(0) & 0xFF == 27:
            break
