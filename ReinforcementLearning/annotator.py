import cv2 as cv
import random
from os import listdir
from os.path import isfile, join

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

    def perform_action(self, x1: float, y1: float, x2: float, y2: float):
        if (x1 < 0 or x1 > 1) or (x2 < 0 or x2 > 1) or\
            (y1 < 0 or y1 > 1) or (y2 < 0 or y2 > 1):
            return
        
        if (x1 > x2):
            x1, x2 = x2, x1

        if (y1 > y2):
            y1, y2 = y2, y1
        
        self.img[int(y1*self.height):int(y2*self.height), int(x1*self.width):int(x2*self.width)] = 255

    def render(self):
        cv.imshow("lol", self.img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
        
if __name__ == "__main__":
    annotator = Annotator()
    annotator.render()

    for i in range(20):
        x1, y1, x2, y2 = random.random(), random.random(), random.random(), random.random()
        print(x1, y1, x2, y2)
        annotator.perform_action(x1, y1, x2, y2)
        annotator.render()
