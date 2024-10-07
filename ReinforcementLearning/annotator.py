import cv2 as cv
import random
from os import listdir
from os.path import isfile, join
import numpy as np

class Annotator:
    def __init__(self, height: int = 270, width: int = 480, folder: str = "images") -> None:
        self.height = height
        self.width = width
        self.img = None
        self.files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
        self.reset()

    def reset(self, seed=None):
        random.seed(seed)
        self.img = cv.imread(random.choice(self.files), cv.IMREAD_COLOR)
        self.img = cv.resize(self.img, (self.width, self.height))
        self.img = cv.Canny(self.img, 25, 50)
        self.img = np.expand_dims(self.img, axis=0)

    def perform_action(self, x: float, y: float, w: float, h: float):
        if (x < 0 or x > 1) or (w < 0 or w > 1) or\
            (y < 0 or y > 1) or (h < 0 or h > 1):
            return
        
        if (x + w/2 > 1 or x < w/2):
            w = min(x*2, (1-x)*2)

        if (y + h/2 > 1 or y < h/2):
            h = min(y*2, (1-y)*2)
        
        self.img[0, int((y-h/2)*self.height):int((y+h/2)*self.height), int((x-w/2)*self.width):int((x+w/2)*self.width)] = 255

    def render(self):
        cv.imshow("lol", self.img[0])
        cv.waitKey(0)
        cv.destroyAllWindows()
    
        
if __name__ == "__main__":
    annotator = Annotator()
    annotator.render()

    for i in range(20):
        x, y, w, h = random.random(), random.random(), random.random(), random.random()
        print(x, y, w, h)
        annotator.perform_action(x, y, w, h)
        annotator.render()
