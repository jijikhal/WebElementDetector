RectF = tuple[float, float, float, float]
RectI = tuple[int, int, int, int]

class BoundingBox:
    """
    Class for representing bounding boxes
    - x, y are coordinates of the top-left corner
    - w, h represent width and height of the bounding box

    All are in relative coordinates (0-1)
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
        """
        Returns a bounding box in the following format:
        (X of middle, Y of middle, Width, Height)
        """
        return (self.x+self.w/2, self.y+self.h/2, self.w, self.h)

    def get_bb_tl(self) -> RectF:
        """
        Returns a bounding box in the following format:
        (X of top-left, Y of top-left, Width, Height)
        """
        return (self.x, self.y, self.w, self.h)
    
    def get_rect(self, img_width: int, img_height: int) -> RectI:
        """
        Returns a bounding box in absolute coordinates int the TL format.
        """
        return (int(self.x*img_width), int(self.y*img_height), int(self.w*img_width), int(self.h*img_height))

    def is_fully_inside(self, other: 'BoundingBox') -> bool:
        """
        Returns whether `other` bounding box is completaly within this bounding box.
        """
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