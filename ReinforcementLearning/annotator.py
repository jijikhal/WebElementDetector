import cv2 as cv
import random
from os import listdir
from os.path import isfile, join
import numpy as np
from cv2.typing import MatLike

RectF = tuple[float, float, float, float]
RectI = tuple[int, int, int, int]

IMG_WIDTH = 100
IMG_HEIGHT = 100

class BoundingBox:
    """
    Class for representing bounding boxes
    - x, y are coordinates of the top-left corner
    - w, h represent width and height of the bounding box

    All are in relative coordinates
    """
    def __init__(self, bb: RectF, middle: bool = True) -> None:
        x, y, w, h = bb
        if (middle):
            if (x + w/2 > 1 or x < w/2):
                w = min(x*2, (1-x)*2)
            if (y + h/2 > 1 or y < h/2):
                h = min(y*2, (1-y)*2)
            self.x: float = x-w/2
            self.y: float = y-h/2
        else:
            if (x+w > 1):
                w = 1-x
            if (y+h > 1):
                h = 1-y
            self.x: float = x
            self.y: float = y
        self.w: float = w
        self.h: float = h

    def get_bb_middle(self) -> RectF:
        return (self.x+self.w/2, self.y+self.h/2, self.w, self.h)

    def get_bb_tl(self) -> RectF:
        return (self.x, self.y, self.w, self.h)
    
    def get_rect(self) -> RectI:
        return (int(self.x*IMG_WIDTH), int(self.y*IMG_HEIGHT), int(self.w*IMG_WIDTH), int(self.h*IMG_WIDTH))

    def is_fully_inside(self, other: 'BoundingBox') -> bool:
        x1, y1, w1, h1 = self.get_bb_tl()
        x2, y2, w2, h2 = other.get_bb_tl()

        return x1 <= x2 and y1 <= y2 and x1+w1 >= x2+w2 and y1+h1 >= y2+h2
    
    def is_intersecting(self, other: 'BoundingBox') -> bool:
        x1, y1, w1, h1 = self.get_bb_tl()
        x2, y2, w2, h2 = other.get_bb_tl()

        if x1 + w1 <= x2 or x2 + w2 <= x1:
            return False
        if y1 + h1 <= y2 or y2 + h2 <= y1:
            return False
        
        if (self.is_fully_inside(other) or other.is_fully_inside(self)):
            return False
        
        return True
    
    def distance(self, other: 'BoundingBox') -> float:
        x1, y1, w1, h1 = self.get_bb_tl()
        x2, y2, w2, h2 = other.get_bb_tl()
        return min(abs(x1-x2), abs(y1-y2), abs(x1+w1-x2-w2), abs(y1+h1-y2-h2))
    
    def intersection_over_union(self, other: 'BoundingBox') -> float:
        x1, y1, w1, h1 = self.get_bb_tl()
        x2, y2, w2, h2 = other.get_bb_tl()
        x, y = max(x1, x2), max(y1, y2)
        xd, yd = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
        overlap = max(0, xd-x)*max(0, yd-y)
        union = w1*h1 + w2*h2 - overlap

        if (union <= 0):
            return 0
        
        return overlap/union
        

class Annotator:
    def __init__(self, height: int, width: int, folder: str = "images") -> None:
        global IMG_HEIGHT
        global IMG_WIDTH
        IMG_HEIGHT = height
        IMG_WIDTH = width
        self.img: MatLike = None
        self.files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
        self.reset()

    def reset(self, seed=None):
        random.seed(seed)
        self.img = cv.imread(random.choice(self.files), cv.IMREAD_COLOR)
        self.img = cv.resize(self.img, (IMG_WIDTH, IMG_HEIGHT))
        self.img = cv.Canny(self.img, 25, 50)

        self.img = np.expand_dims(self.img, axis=0) # convert to channel first


    def perform_action(self, action: RectF) -> None:
        x, y, w, h = action
        assert not ((x < 0 or x > 1) or (w < 0 or w > 1) or (y < 0 or y > 1) or (h < 0 or h > 1))

        if (x + w/2 > 1 or x < w/2):
            w = min(x*2, (1-x)*2)

        if (y + h/2 > 1 or y < h/2):
            h = min(y*2, (1-y)*2)

        bb = BoundingBox((x,y,w,h), True)

        cv.rectangle(self.img[0], bb.get_rect(), (255,), -1)
    

    def render(self):
        cv.imshow("lol", self.img[0])
        #cv.waitKey(0)
        #cv.destroyAllWindows()
    
        
if __name__ == "__main__":
    annotator = Annotator(100, 100)
    annotator.render()

    for i in range(20):
        x, y, w, h = random.random(), random.random(), random.random(), random.random()
        print(x, y, w, h)
        annotator.perform_action((x, y, w, h))
        annotator.render()
