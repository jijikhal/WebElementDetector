import cv2
from cv2.typing import MatLike
from os import listdir
from os.path import isfile, join
from typing import Callable, Any, Sequence, cast
from time import time
import numpy as np

from utils.draw_bb import draw_bounding_boxes
from wed.bounding_box import BoundingBox, BoundingBoxType


class NodeBoundingBoxBased():
    def __init__(self, bb: BoundingBox, contour: MatLike) -> None:
        self.bb = bb
        self.contour = contour
        self.parent: NodeBoundingBoxBased | None = None
        self.children: list[NodeBoundingBoxBased] = []
        self.root = False

    def add_node(self, node: 'NodeBoundingBoxBased'):
        for c in self.children:
            if c.bb.percentage_inside(node.bb) > 0.9:
                c.add_node(node)
                return

        if self.bb.iou(node.bb) > 0.8:
            return
        self.children.append(node)
        node.parent = self

    def total_child_count(self) -> int:
        total = len(self.children)
        for c in self.children:
            total += c.total_child_count()
        return total

    def run_function(self, f: Callable[['NodeBoundingBoxBased'], Any]) -> None:
        f(self)
        for c in self.children:
            c.run_function(f)

    def filter_nodes(self, f: Callable[['NodeBoundingBoxBased'], bool], edged_image: MatLike) -> None:
        should_be_discarded = f(self)
        if (should_be_discarded and self.parent is not None):
            for c in self.children.copy():
                c.filter_nodes(lambda x: True, edged_image)
            self.parent.children.remove(self)
            cv2.drawContours(
                edged_image, [self.contour], -1, (0,), cv2.FILLED)
            return

        for c in self.children.copy():
            c.filter_nodes(f, edged_image)

    def get_bbs(self, result: list[BoundingBox] | None = None) -> list[BoundingBox]:
        if result is None:
            result = []

        result.append(self.bb)
        for c in self.children:
            c.get_bbs(result)

        return result


def make_bb_tree(contours: Sequence[MatLike], img_w: int, img_h: int) -> NodeBoundingBoxBased:
    nodes = [NodeBoundingBoxBased(BoundingBox(cv2.boundingRect(
        c), BoundingBoxType.OPEN_CV, img_w, img_h), c) for c in contours]
    nodes.sort(key=lambda x: x.bb.area(), reverse=True)

    root = NodeBoundingBoxBased(BoundingBox(
        (0, 0, img_w, img_h), BoundingBoxType.OPEN_CV, img_w, img_h), cast(MatLike, np.array([[[0, 0]], [[img_w-1, 0]], [[img_w-1, img_h-1]], [[0, img_h-1]]])))
    if len(nodes) > 0 and nodes[0].bb.iou(root.bb) > 0.98:
        root = nodes[0]
    root.root = True
    for n in nodes:
        if n.root:
            continue
        root.add_node(n)

    return root


def contour_inside(contour: MatLike, edged_image: MatLike) -> bool:
    mask: MatLike = np.zeros_like(edged_image)
    cv2.drawContours(mask, [contour], -1, (255,), thickness=cv2.FILLED)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    count_white = np.count_nonzero(cv2.bitwise_and(edged_image, mask))

    return count_white == 0


def remove_bridges(img: MatLike) -> None:
    # Create a copy and convert to 0 and 1
    img[img == 255] = 1

    kernel_h = np.array([
        [0,  1,  0],
        [-1, 1, -1],
        [0,  1,  0]
    ], dtype=np.int8)

    kernel_v = np.array([
        [0, -1, 0],
        [1,  1, 1],
        [0, -1, 0]
    ], dtype=np.int8)

    hitmiss_h = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel_h)
    hitmiss_v = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel_v)

    bridges = cv2.bitwise_or(hitmiss_h, hitmiss_v)
    img[bridges == 1] = 0
    img[img == 1] = 255


def find_elements_cv(img: MatLike, include_root: bool = True) -> tuple[list[BoundingBox], MatLike]:
    img_h, img_w, _ = img.shape

    canny = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.bilateralFilter(canny, 21, 50, 10)
    canny = cv2.Canny(canny, 100, 200, L2gradient=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    canny = cv2.morphologyEx(canny, cv2.MORPH_DILATE, kernel)
    remove_bridges(canny)

    contours, _ = cv2.findContours(
        canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    root = make_bb_tree(contours, img_w, img_h)

    # Flter out too small elements
    root.filter_nodes(lambda x: max(x.bb.abs_width(),
                      x.bb.abs_height()) < 30, canny)
    root.filter_nodes(lambda x: min(
        x.bb.abs_width(), x.bb.abs_height()) < 15, canny)
    # Small elements can not have children
    root.filter_nodes(lambda x: x.parent is not None and (
        x.parent.bb.abs_height() < 100 and x.parent.bb.abs_width() < 100), canny)
    # Filter out hole elements
    root.filter_nodes(lambda x: 10 < cv2.minAreaRect(x.contour)[2] < 80 and not (0.5 < x.bb.aspect_ratio() < 1.5), canny)
    root.filter_nodes(lambda x: cv2.contourArea(cv2.convexHull(x.contour), False) / x.bb.abs_area() < 0.75 and 2.5 < cv2.minAreaRect(x.contour)[2] < 87.5, canny)
    root.filter_nodes(lambda x: contour_inside(x.contour, canny), canny)

    bbs = root.get_bbs()
    merged: list[BoundingBox] = []
    used: set[BoundingBox] = set()
    bbs.sort(key=lambda x: x.area())
    for b in bbs:
        if (b in used):
            continue
        used.add(b)
        to_merge_with = [
            x for x in bbs if b.is_intersecting(x) and x not in used]
        m = b
        for t in to_merge_with:
            used.add(t)
            m = b.merge(t)
        merged.append(m)
    
    if not include_root:
        merged.sort(key=lambda x: x.area(), reverse=True)
        return merged[1:], canny

    return merged, canny

if __name__ == "__main__":
    dataset_folder = r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\wed\rl\dataset_big"
    paths = [join(dataset_folder, f) for f in listdir(
        dataset_folder) if isfile(join(dataset_folder, f))]
    for p in paths:
        img = cv2.imread(p)
        img_copy = img.copy()
        start = time()
        boxes, _ = find_elements_cv(img)
        print(time()-start)
        draw_bounding_boxes(img_copy, boxes, (255, 255, 0))
        #cv2.imshow("before", img_copy)
        #if chr(cv2.waitKey(0)) == 'q':
        #    break
