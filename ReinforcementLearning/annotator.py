import cv2 as cv
import random
from os import listdir
from os.path import isfile, join
import numpy as np
class BoundingBox:
    def __init__(self, x, y, w, h) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def get_top_left(self):
        return (self.x-self.w/2, self.y-self.h/2, self.w, self.h)
    
    def get_rect(self, w_img, h_img):
        x, y, w, h = self.get_top_left()
        return (int(x*w_img), int(y*h_img), int(w*w_img), int(h*h_img))
        
    
    def set_top_left(self, x, y, w, h):
        self.x = x+w/2
        self.y = y+h/2
        self.w = w
        self.h = h

    def is_fully_inside(self, other: 'BoundingBox'):
        x1, y1, w1, h1 = self.get_top_left()
        x2, y2, w2, h2 = other.get_top_left()

        return x1 <= x2 and y1 <= y2 and x1+w1 >= x2+w2 and y1+h1 >= y2+h2
    
    def is_intersecting(self, other: 'BoundingBox'):
        x1, y1, w1, h1 = self.get_top_left()
        x2, y2, w2, h2 = other.get_top_left()

        if x1 + w1 <= x2 or x2 + w2 <= x1:
            return False
        if y1 + h1 <= y2 or y2 + h2 <= y1:
            return False
        
        if (self.is_fully_inside(other) or other.is_fully_inside(self)):
            return False
        
        return True
    
    def distance(self, other: 'BoundingBox'):
        x1, y1, w1, h1 = self.get_top_left()
        x2, y2, w2, h2 = other.get_top_left()
        return min(abs(x1-x2), abs(y1-y2), abs(x1+w1-x2-w2), abs(y1+h1-y2-h2))
        

class Annotator:
    def __init__(self, height: int = 100, width: int = 100, folder: str = "images") -> None:
        self.height = height
        self.width = width
        self.img = None
        self.contours: list[BoundingBox] = []
        self.files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
        self.reset()

    def reset(self, seed=None):
        random.seed(seed)
        self.img = cv.imread(random.choice(self.files), cv.IMREAD_COLOR)
        self.img = cv.resize(self.img, (self.width, self.height))
        self.img = cv.Canny(self.img, 25, 50)

        contours, _ = cv.findContours(self.img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv.boundingRect(c)
            bb = BoundingBox(0, 0, 0, 0)
            bb.set_top_left(x/self.width, y/self.height, w/self.width, h/self.height)
            self.contours.append(bb)
            #cv.rectangle(self.img, (x, y), (x+w, y+h), 127)        

        self.img = np.expand_dims(self.img, axis=0) # convert to channel first


    def perform_action(self, x: float, y: float, w: float, h: float):
        if (x < 0 or x > 1) or (w < 0 or w > 1) or\
            (y < 0 or y > 1) or (h < 0 or h > 1):
            return
        
        if (x + w/2 > 1 or x < w/2):
            w = min(x*2, (1-x)*2)

        if (y + h/2 > 1 or y < h/2):
            h = min(y*2, (1-y)*2)

        bb = BoundingBox(x,y,w,h)

        """debug_img = self.img[0].copy()
        debug_img = cv.cvtColor(debug_img, cv.COLOR_GRAY2BGR)
        cv.rectangle(debug_img, bb.get_rect(self.width, self.height), (0, 255, 0))"""

        
        inside: list[BoundingBox] = list(filter(lambda x: bb.is_fully_inside(x), self.contours))
        colliding: list[BoundingBox] = list(filter(lambda x: bb.is_intersecting(x), self.contours))

        """for c in inside:
            cv.rectangle(debug_img, c.get_rect(self.width, self.height), (255, 0, 0))

        for c in colliding:
            cv.rectangle(debug_img, c.get_rect(self.width, self.height), (0, 0, 255))

        cv.imshow("lol", debug_img)
        cv.waitKey(0)
        cv.destroyAllWindows()"""

        reward = 0
        reward -= len(colliding)
        reward += len(list(filter(lambda x: bb.distance(x) <= 0.05, inside)))

        for c in inside:
            self.contours.remove(c)

        self.contours.append(bb)
        
        self.img[0, int((y-h/2)*self.height):int((y+h/2)*self.height), int((x-w/2)*self.width):int((x+w/2)*self.width)] = 255

        return reward

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
        print(annotator.perform_action(x, y, w, h))
        annotator.render()
