from enum import Enum
from math import exp, sqrt
from cv2.typing import Rect

RectF = tuple[float, float, float, float]
RectI = tuple[int, int, int, int]

class BoundingBoxType(Enum):
    CENTER = 1
    TOP_LEFT = 2
    TWO_CORNERS = 3
    OPEN_CV = 4

class BoundingBox:
    """
    Class for representing bounding boxes
    - x, y are coordinates of the top-left corner
    - w, h represent width and height of the bounding box

    All are in relative coordinates (0-1)
    """
    def __init__(self, bb: RectF | list[float] | Rect, type: BoundingBoxType = BoundingBoxType.CENTER, img_w: int = -1, img_h: int = -1) -> None:
        self.x: float
        self.y: float
        self.w: float
        self.h: float

        self.img_w: int = img_w
        self.img_h: int = img_h
        self.is_absolute: bool = False

        if type == BoundingBoxType.CENTER:
            x, y, w, h = bb
            if (x + w/2 > 1 or x < w/2):
                w = min(x*2, (1-x)*2)
            if (y + h/2 > 1 or y < h/2):
                h = min(y*2, (1-y)*2)
            self.x = x-w/2
            self.y = y-h/2
            self.w = w
            self.h = h
        elif type == BoundingBoxType.TOP_LEFT:
            x, y, w, h = bb
            if (x+w > 1):
                w = 1-x
            if (y+h > 1):
                h = 1-y
            self.x = x
            self.y = y
            self.w = w
            self.h = h
        elif type == BoundingBoxType.TWO_CORNERS:
            x1, y1, x2, y2 = bb
            self.x = min(x1, x2)
            self.y = min(y1, y2)
            self.w = abs(x1-x2)
            self.h = abs(y1-y2)
        elif type == BoundingBoxType.OPEN_CV:
            assert img_h > 0 and img_w > 0, "For OpenCv type the image size must be specified"
            self.is_absolute = True
            x_abs, y_abs, w_abs, h_abs = bb
            x = x_abs/img_w
            y = y_abs/img_h
            w = w_abs/img_w
            h = h_abs/img_h
            if (x+w > 1):
                w = 1-x
            if (y+h > 1):
                h = 1-y
            self.x = x
            self.y = y
            self.w = w
            self.h = h

    def abs_width(self) -> int:
        assert self.is_absolute, "absolute width is only available for absolut defined bounding boxes"
        return round(self.w*self.img_w)

    def abs_height(self) -> int:
        assert self.is_absolute, "absolute height is only available for absolut defined bounding boxes"
        return round(self.h*self.img_h)
    
    def abs_area(self) -> int:
        return self.abs_height()*self.abs_width()


    def __repr__(self) -> str:
        return f"<BoundingBox: ({self.x}, {self.y}, {self.w}, {self.h})>"

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
    
    def get_bb_corners(self) -> RectF:
        return (self.x, self.y, self.x+self.w, self.y+self.h)
    
    def get_rect(self, img_width: int = -1, img_height: int = -1) -> RectI:
        """
        Returns a bounding box in absolute coordinates int the TL format.
        """
        assert self.is_absolute or (img_height > 0 and img_width > 0), "For non-OpenCV bounding boxes, img size must be specified"
        if (self.is_absolute and (img_height < 0 or img_width < 0)):
            img_height = self.img_h
            img_width = self.img_w
        return (round(self.x*img_width), round(self.y*img_height), round(self.w*img_width), round(self.h*img_height))

    def fully_contains(self, other: 'BoundingBox') -> bool:
        """
        Returns whether `other` bounding box is completaly within this bounding box.
        """

        if self is other:
            return False

        x1, y1, w1, h1 = self.get_bb_tl()
        x2, y2, w2, h2 = other.get_bb_tl()

        if (abs(x1-x2) < 0.01 and abs(y1-y2) < 0.01 and abs(w1-w2) < 0.01 and abs(h1-h2) < 0.01):
            return False

        return x1 <= x2 and y1 <= y2 and x1+w1 >= x2+w2 and y1+h1 >= y2+h2
    
    def area(self) -> float:
        _, _, w, h = self.get_bb_tl()
        return w*h
    
    def is_intersecting(self, other: 'BoundingBox') -> bool:
        """
        Returns whether the bounding rectangles of the two bounding boxes have any common points.

        Importantly if one is fully inside the other, it returns False.
        """
        x1, y1, w1, h1 = self.get_bb_tl()
        x2, y2, w2, h2 = other.get_bb_tl()

        if x1 + w1 <= x2 or x2 + w2 <= x1:
            return False
        if y1 + h1 <= y2 or y2 + h2 <= y1:
            return False
        
        if (self.fully_contains(other) or other.fully_contains(self)):
            return False
        
        return True
    
    def overlap(self, other: 'BoundingBox') -> float:
        """
        Returns the area of an overlap of two bounding boxes
        """
        x1, y1, w1, h1 = self.get_bb_tl()
        x2, y2, w2, h2 = other.get_bb_tl()
        x, y = max(x1, x2), max(y1, y2)
        xd, yd = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
        return max(0, xd-x)*max(0, yd-y)
    
    def percentage_inside(self, other: 'BoundingBox') -> float:
        return self.overlap(other) / other.area()
    
    def has_overlap(self, other: 'BoundingBox') -> bool:
        """
        Returns whether two bounding boxes have any overlap
        """
        return self.overlap(other) > 0
    
    def iou(self, other: 'BoundingBox') -> float:
        """
        Calculates the IoU metric (Jaccard index) for two bounding boxes
        """
        x1, y1, w1, h1 = self.get_bb_tl()
        x2, y2, w2, h2 = other.get_bb_tl()
        x, y = max(x1, x2), max(y1, y2)
        xd, yd = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
        overlap = max(0, xd-x)*max(0, yd-y)
        union = w1*h1 + w2*h2 - overlap

        if (union <= 0):
            return 0
        
        return overlap/union
    
    @staticmethod
    def _shift_closer_by_tolerance(x: float, y: float, tolerance: float) -> tuple[float, float]:
        if abs(x-y) < tolerance:
            result = (x+y)/2
            return result, result
        elif x < y:
            return x+tolerance/2, y-tolerance/2
        else:
            return x-tolerance/2, y+tolerance/2

    def tolerant_iou(self, other: 'BoundingBox', tolerance: float = 0.01, weight_of_modified: float = 0.75) -> float:
        assert 0 <= tolerance <= 1, "Tolerance must be 0-1"
        assert 0 <= weight_of_modified <= 1, "Tolerance must be 0-1"

        iou_before = self.iou(other)

        x1, y1, x2, y2 = self.get_bb_corners()
        x3, y3, x4, y4 = other.get_bb_corners()

        x1, x3 = BoundingBox._shift_closer_by_tolerance(x1, x3, tolerance)
        x2, x4 = BoundingBox._shift_closer_by_tolerance(x2, x4, tolerance)
        y1, y3 = BoundingBox._shift_closer_by_tolerance(y1, y3, tolerance)
        y2, y4 = BoundingBox._shift_closer_by_tolerance(y2, y4, tolerance)

        new_bb_1 = BoundingBox((x1, y1, x2, y2), BoundingBoxType.TWO_CORNERS)
        new_bb_2 = BoundingBox((x3, y3, x4, y4), BoundingBoxType.TWO_CORNERS)

        iou_after = new_bb_1.iou(new_bb_2)
        return iou_before*(1-weight_of_modified) + iou_after*weight_of_modified
    
    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1 / (1 + exp(-x))
    
    def ASS(self, other: 'BoundingBox', sharpness: float = 2.8, shift: float = 0.5) -> float:
        """
        Computes the Absolute Sigmoid Score
        """
        x1, y1, x2, y2 = self.get_bb_corners()
        x3, y3, x4, y4 = other.get_bb_corners()

        cx1, cy1 = (x1+x2)/2, (y1+y2)/2
        cx2, cy2 = (x3+x4)/2, (y3+y4)/2
        diff = abs(x1-x3) + abs(x2-x4) + abs(y1-y3) + abs(y2-y4) + abs(cx1-cx2) + abs(cy1-cy2)

        score = 1 - BoundingBox._sigmoid(sharpness * (diff - shift))
        norm_factor = 1 - BoundingBox._sigmoid(-sharpness * shift)

        return score / norm_factor


    def get_distance(self, other: 'BoundingBox') -> float:
        x1, y1, x2, y2 = self.get_bb_corners()
        x3, y3, x4, y4 = other.get_bb_corners()

        # Generated by ChatGPT:
        # Check for overlap in the x-direction
        if x2 < x3:
            dx = x3 - x2
        elif x4 < x1:
            dx = x1 - x4
        else:
            dx = 0

        # Check for overlap in the y-direction
        if y2 < y3:
            dy = y3 - y2
        elif y4 < y1:
            dy = y1 - y4
        else:
            dy = 0

        distance = sqrt(dx**2 + dy**2)
        return distance
    
    def merge(self, other: 'BoundingBox') -> 'BoundingBox':
        x1, y1, x2, y2 = self.get_bb_corners()
        x3, y3, x4, y4 = other.get_bb_corners()

        result = BoundingBox((min(x1, x3), min(y1, y3), max(x2, x4), max(y2, y4)), BoundingBoxType.TWO_CORNERS)
        if (self.is_absolute and other.is_absolute and self.img_w == other.img_w and self.img_h == other.img_h):
            result.is_absolute = True
            result.img_h = self.img_h
            result.img_w = self.img_w

        return result

