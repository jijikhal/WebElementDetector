import cv2
from cv2.typing import MatLike
from os import listdir
from os.path import isfile, join
from typing import Callable, Any, Sequence
from time import time
import numpy as np

from bounding_box import BoundingBox, BoundingBoxType


class NodeBoundingBoxBased():
    def __init__(self, bb: BoundingBox, contour: MatLike | None) -> None:
        self.bb = bb
        self.contour = contour
        self.parent: NodeBoundingBoxBased | None = None
        self.children: list[NodeBoundingBoxBased] = []
        self.image = False
        self.root = False

    def add_node(self, node: 'NodeBoundingBoxBased'):
        for c in self.children:
            if c.bb.percentage_inside(node.bb) > 0.9:
                c.add_node(node)
                return

        if self.bb.iou(node.bb) > 0.9:
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
            if self.contour is not None:
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
        (0, 0, img_w, img_h), BoundingBoxType.OPEN_CV, img_w, img_h), None)
    if (nodes[0]).bb.iou(root.bb) > 0.98:
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


def find_elements_cv(img: MatLike):
    img_h, img_w, _ = img.shape
    copy_for_show = img.copy()

    start = time()
    canny = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.bilateralFilter(canny, 21, 50, 10)
    canny = cv2.Canny(canny, 22, 22, L2gradient=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    canny = cv2.morphologyEx(canny, cv2.MORPH_DILATE, kernel)
    remove_bridges(canny)

    cv2.imshow("edged", canny)

    contours, _ = cv2.findContours(
        canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("Preprocessing took:", time()-start)

    start = time()
    root = make_bb_tree(contours, img_w, img_h)
    print("tree creation took:", time()-start)

    start = time()
    # Flter out too small elements
    root.filter_nodes(lambda x: max(x.bb.abs_width(), x.bb.abs_height()) < 30 or min(
        x.bb.abs_width(), x.bb.abs_height()) < 10, canny)
    # Small elements can not have children
    root.filter_nodes(lambda x: x.parent is not None and (
        x.parent.bb.abs_height() < 100 and x.parent.bb.abs_width() < 100), canny)
    # Filter out hole elements
    root.filter_nodes(lambda x: x.contour is not None and contour_inside(
        x.contour, canny), canny)

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

    print("filtering took:", time()-start)

    for b in merged:
        cv2.rectangle(copy_for_show, b.get_rect(), (255, 255, 0), 2)

    cv2.imshow("lol", copy_for_show)
    cv2.waitKey(0)


if __name__ == "__main__":
    dataset_folder = r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\TraditionalCV\testing_data"
    paths = [join(dataset_folder, f) for f in listdir(
        dataset_folder) if isfile(join(dataset_folder, f))]
    for p in paths:
        img = cv2.imread(p)
        find_elements_cv(img)
