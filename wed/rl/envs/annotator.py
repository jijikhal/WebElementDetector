import cv2 as cv
import random
from os import listdir
from os.path import isfile, join
import numpy as np
from cv2.typing import MatLike
from wed.bounding_box import BoundingBox, RectF, RectI, BoundingBoxType

class Annotator:
    def __init__(self, height: int, width: int, folder: str = "images") -> None:
        global IMG_HEIGHT
        global IMG_WIDTH
        IMG_HEIGHT = height
        IMG_WIDTH = width
        self.img: MatLike
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

        bb = BoundingBox((x,y,w,h), BoundingBoxType.CENTER)

        cv.rectangle(self.img[0], bb.get_rect(self.img[0].shape[1], self.img[0].shape[0]), (255,), -1)
    

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
