from bounding_box import BoundingBox, BoundingBoxType
import cv2
from cv2.typing import MatLike
from random import uniform, choice, randint, seed
import numpy as np

MIN_SIZE = 0.02 # minimal size of object
MIN_SPACING = 0.03 # minimal distance between two objects
MAX_SIBILINGS = 4 # max amount of children
MAX_DEPTH = 4 # max depth
MAX_EDGE = 0.2 # max space from parent to child

class Node:
    def __init__(self, bb: BoundingBox, level: int, parent: 'Node | None') -> None:
        self.bb = bb
        self.children: list[Node] = []
        self.level = level
        self.parent = parent

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def draw_self(self, img: MatLike) -> MatLike:
        height, width = img.shape
        x1, y1, x2, y2 = self.bb.get_bb_corners()
        cv2.rectangle(img, (int(x1*width), int(y1*height)), (int(x2*width)-1, int(y2*height)-1), (255,), 1)
        for c in self.children:
            c.draw_self(img)
        return img

    def create_children(self, hierarchy: list['Node']) -> None:
        hierarchy.append(self)
        if self.level >= MAX_DEPTH:
            return
        x, y, w, h = self.bb.get_bb_tl()
        if (w*MAX_EDGE < MIN_SPACING or h*MAX_EDGE < MIN_SPACING):
            return
        l_edge = uniform(MIN_SPACING, MAX_EDGE*w)
        r_edge = uniform(MIN_SPACING, MAX_EDGE*w)
        t_edge = uniform(MIN_SPACING, MAX_EDGE*h)
        b_edge = uniform(MIN_SPACING, MAX_EDGE*h)

        if (l_edge+r_edge < MIN_SIZE or t_edge+b_edge < MIN_SIZE):
            return

        child_space = BoundingBox((x+l_edge, y+t_edge, x+w-r_edge, y+h-b_edge), BoundingBoxType.TWO_CORNERS)
        cx, cy, cw, ch = child_space.get_bb_tl()

        horizontal = choice([True, False])

        spacing = uniform(MIN_SPACING, 5*MIN_SPACING)

        max_allowed_children = min(int(cw/(MIN_SIZE+spacing) if horizontal else ch/(MIN_SIZE+spacing)), MAX_SIBILINGS)
        if (max_allowed_children == 0):
            return
        
        child_count = randint(1 if self.level == 0 else 0, max_allowed_children)
        if (child_count == 0):
            return
        child_sizes = [randint(10, 100) for _ in range(child_count)]
        coef = ((cw if horizontal else ch)-(child_count-1)*spacing)/sum(child_sizes)
        child_sizes = list(map(lambda x: x*coef, child_sizes))
        for i in range(child_count):
            if horizontal:
                child_bb = BoundingBox((cx+sum(child_sizes[:i])+i*spacing, cy, child_sizes[i], ch), BoundingBoxType.TOP_LEFT)
            else:
                child_bb = BoundingBox((cx, cy+sum(child_sizes[:i])+i*spacing, cw, child_sizes[i]), BoundingBoxType.TOP_LEFT)
            child = Node(child_bb, self.level+1, self)
            self.children.append(child)
            child.create_children(hierarchy)

    def remove(self, img: MatLike, hierarchy: list['Node'], remove_from_parent=True):
        if (self.parent is not None and remove_from_parent):
            self.parent.children.remove(self)

        height, width = img.shape
        x1, y1, x2, y2 = self.bb.get_bb_corners()
        cv2.rectangle(img, (int(x1*width), int(y1*height)), (int(x2*width)-1, int(y2*height)-1), (0,), -1)
        hierarchy.remove(self)
        for i in self.children:
            i.remove(img, hierarchy, False)

def generete_hierarchy(size: tuple[int, int], seed_set = None) -> tuple[MatLike, list[Node]]:
    seed(seed_set)
    root = Node(BoundingBox((0,0,1,1), BoundingBoxType.TOP_LEFT), 0, None)
    img = np.zeros(size, dtype=np.uint8)
    hierarchy: list[Node] = []
    root.create_children(hierarchy)
    root.draw_self(img)
    return img, hierarchy

if __name__ == "__main__":
    total = 0
    for _ in range(10000):
        img, h = generete_hierarchy((100, 100), None)
        total += len(h)
    print(total/10000)

    """while True:
        img, _ = generete_hierarchy((100, 100), None)
        cv2.imshow("hierarchy", img)
        if cv2.waitKey(0) & 0xFF == 27:
            break
    cv2.destroyAllWindows()"""
