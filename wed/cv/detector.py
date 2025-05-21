# This file contains the implementation of purly Traditional CV-based detector
# from Section 5.1 and Appendix B

import cv2
from cv2.typing import MatLike
from os import listdir
from os.path import isfile, join
from typing import Callable, Any, Sequence, cast
from time import time
import numpy as np

from wed.utils.draw_bb import draw_bounding_boxes
from utils.bounding_box import BoundingBox, BoundingBoxType
from wed.utils.get_files_in_folder import get_files


class Node():
    """Class for the reprezentation of a hierarchy of elements.
    """

    def __init__(self, bb: BoundingBox, contour: MatLike) -> None:
        """
        Args:
            bb (BoundingBox): Enclosing axis-aligned bounding box of the node.
            contour (MatLike): The contour represented by this Node.
        """
        self.bb = bb
        self.contour = contour
        self.parent: Node | None = None
        self.children: list[Node] = []
        self.root = False

    def add_node(self, node: 'Node') -> None:
        """Adds a child node to this node.

        Args:
            node (Node): The node to be added as a child.
        """
        for c in self.children:
            if c.bb.percentage_inside(node.bb) > 0.9:
                c.add_node(node)
                return

        if self.bb.iou(node.bb) > 0.8:
            return
        self.children.append(node)
        node.parent = self

    def total_child_count(self) -> int:
        """Returns the total amount of children (including indirect).

        Returns:
            int: The total child count.
        """
        total = len(self.children)
        for c in self.children:
            total += c.total_child_count()
        return total

    def run_function(self, f: Callable[['Node'], Any]) -> None:
        """Runs the provided function on self and every child recursivly.

        Args:
            f (Callable[[&#39;Node&#39;], Any]): The function to be run.
        """
        f(self)
        for c in self.children:
            c.run_function(f)

    def filter_nodes(self, f: Callable[['Node'], bool], edged_image: MatLike) -> None:
        """Filters out nodes that fulfill the provided predicate and removes their contour from the image.

        Args:
            f (Callable[[&#39;Node&#39;], bool]): The predicate the node has to fulfill to be removed.
            edged_image (MatLike): Image to remove the contours from.
        """
        should_be_discarded = f(self)
        if (should_be_discarded and self.parent is not None):
            for c in self.children.copy():
                c.filter_nodes(lambda _: True, edged_image)
            self.parent.children.remove(self)
            cv2.drawContours(
                edged_image, [self.contour], -1, (0,), cv2.FILLED)
            return

        for c in self.children.copy():
            c.filter_nodes(f, edged_image)

    def get_bbs(self, result: list[BoundingBox] | None = None) -> list[BoundingBox]:
        """Returns all bounding boxes in the tree where this node is the root. In pre-order order.

        Args:
            result (list[BoundingBox] | None, optional): The list to be used as accumulator. If None, creates new list. Defaults to None.

        Returns:
            list[BoundingBox]: List of all bounding boxes
        """
        if result is None:
            result = []

        result.append(self.bb)
        for c in self.children:
            c.get_bbs(result)

        return result


def make_bb_tree(contours: Sequence[MatLike], img_w: int, img_h: int) -> Node:
    """Creates a hierarchical tree of bounding boxes of the provided contours

    Args:
        contours (Sequence[MatLike]): The contours
        img_w (int): Width of the image from which the contours were extracted in pixel
        img_h (int): Heoght of the image from which the contours were extracted in pixel

    Returns:
        Node: Root node of the tree
    """
    nodes: list[Node] = [Node(BoundingBox(cv2.boundingRect(
        c), BoundingBoxType.OPEN_CV, img_w, img_h), c) for c in contours]
    nodes.sort(key=lambda x: x.bb.area(), reverse=True)

    root = Node(BoundingBox(
        (0, 0, img_w, img_h), BoundingBoxType.OPEN_CV, img_w, img_h), cast(MatLike, np.array([[[0, 0]], [[img_w-1, 0]], [[img_w-1, img_h-1]], [[0, img_h-1]]])))
    if len(nodes) > 0 and nodes[0].bb.iou(root.bb) > 0.98:
        root.contour = nodes[0].contour
    root.root = True
    for n in nodes:
        if n.root:
            continue
        root.add_node(n)

    return root


def contour_inside(contour: MatLike, edged_image: MatLike) -> bool:
    """Determines whether a contour is an inside contour (a contour of a hole) with no children. Not 100 percent reliable

    Args:
        contour (MatLike): The contour to be judged
        edged_image (MatLike): Image in which the contour is

    Returns:
        bool: Whether the contour is an inside contour
    """
    mask: MatLike = np.zeros_like(edged_image)
    cv2.drawContours(mask, [contour], -1, (255,), thickness=cv2.FILLED)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    count_white = np.count_nonzero(cv2.bitwise_and(edged_image, mask))

    return count_white == 0


def remove_bridges(img: MatLike) -> None:
    """Removes one-pixel wide bridges. Made with help of ChatGPT

    Args:
        img (MatLike): The binary image to remove the brideges from
    """
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
    """Finds elements in the provided screenshot of a webpage.

    Args:
        img (MatLike): The image to analyze
        include_root (bool, optional): Whether a root element should be included. Defaults to True.

    Returns:
        tuple[list[BoundingBox], MatLike]: List of all found element bounding boxes and the resulting binary image used.
    """
    img_h, img_w, _ = img.shape
    if (img_h != 900 or img_w != 1440):
        print("Warning: this detector was designed for images of size 1440 x 900 px and might perform worse on different sizes.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, 21, 50, 10)
    canny = cv2.Canny(smooth, 100, 200, L2gradient=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.morphologyEx(canny, cv2.MORPH_DILATE, kernel)
    remove_bridges(dilated)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    root = make_bb_tree(contours, img_w, img_h)

    # Flter out too small elements
    root.filter_nodes(lambda x: max(x.bb.abs_width(), x.bb.abs_height()) < 30, dilated)
    root.filter_nodes(lambda x: min(x.bb.abs_width(), x.bb.abs_height()) < 15, dilated)

    # Small elements can not have children
    root.filter_nodes(lambda x: x.parent is not None and (
        x.parent.bb.abs_height() < 100 and x.parent.bb.abs_width() < 100), dilated)
    # Filter out hole elements
    root.filter_nodes(lambda x: 10 < cv2.minAreaRect(x.contour)[2] < 80 and not (0.5 < x.bb.aspect_ratio() < 1.5), dilated)
    root.filter_nodes(lambda x: cv2.contourArea(cv2.convexHull(x.contour), False) / x.bb.abs_area()
                      < 0.75 and 2.5 < cv2.minAreaRect(x.contour)[2] < 87.5, dilated)
    root.filter_nodes(lambda x: contour_inside(x.contour, dilated), dilated)

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
        return merged[1:], dilated

    return merged, dilated


if __name__ == "__main__":
    dataset_folder = r"rl\dataset_big"
    paths = get_files(dataset_folder)
    for p in paths:
        img = cv2.imread(p)
        img_copy = img.copy()
        start = time()
        boxes, _ = find_elements_cv(img)
        print(time()-start)
        draw_bounding_boxes(img_copy, boxes, (255, 255, 0))
        cv2.imshow("predictions", img_copy)
        if chr(cv2.waitKey(0)) == 'q':
            break
