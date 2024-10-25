import numpy as np
import cv2
from time import time

Rect = tuple[int, int, int, int]

def is_fully_inside(rect1: Rect, rect2: Rect):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    return x1 <= x2 and y1 <= y2 and x1+w1 >= x2+w2 and y1+h1 >= y2+h2

def has_no_child(rect1: Rect, rect2: Rect):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    if (rect1 == rect2):
        return True
    
    if (abs(w1-w2) < 5 or abs(h1-h2) <5):
        return True

    """if x1 + w1 <= x2 or x2 + w2 <= x1:
        return True
    if y1 + h1 <= y2 or y2 + h2 <= y1:
        return True"""
    
    if (not is_fully_inside(rect1, rect2)):
        return True
    
    return False


def calculate_reward(img, rect: Rect):
    start = time()
    total_reward = 0

    ih, iw = img.shape
    x, y, w, h = rect
    cutout = img[y:y+h, x:x+w]
    #cv2.imshow("cutout", cutout)

    # count edge intersections
    edge = cutout.copy()
    edge[1:h-1, 1:w-1] = 0
    total_edge_pixels = w*2+h*2-4
    white_edge_pixels = np.count_nonzero(edge > 0)
    percentage_cross = white_edge_pixels/total_edge_pixels
    print(f"Edge pixels overlap: {white_edge_pixels}, percentage: {percentage_cross}")
    if (white_edge_pixels > 0):
        total_reward -= 100

    # Count enclosed components
    _, binary = cv2.threshold(cutout, 254, 255, cv2.THRESH_BINARY)
    retval, _ = cv2.connectedComponents(binary)
    print(f"Component count: {retval-1}")

    if (5 > retval-1 > 0):
        total_reward += 1
    else:
        total_reward -= 10


    # Count distance from edges
    white_pixel_coords = np.column_stack(np.where(cutout == 255))
    # If there are any white pixels, compute distances
    if white_pixel_coords.size > 0:
        # Get min and max for x and y
        min_y, min_x = white_pixel_coords.min(axis=0)
        max_y, max_x = white_pixel_coords.max(axis=0)

        top_dist = (min_y)/ih
        bottom_dist = (cutout.shape[0] - 1 - max_y)/ih
        left_dist = (min_x)/iw
        right_dist = (cutout.shape[1] - 1 - max_x)/iw

        for d in [top_dist, bottom_dist, left_dist, right_dist]:
            if (d < 0.01):
                total_reward += 1-d*100


        print(f"Distance to closest white pixel from edges:")
        print(f"  Top edge: {top_dist:f}")
        print(f"  Bottom edge: {bottom_dist}")
        print(f"  Left edge: {left_dist}")
        print(f"  Right edge: {right_dist}")

    print(f"Reward is: {total_reward}")
    print(time()-start)
    print(w*h/(iw*ih) > 0.9)

    #cv2.waitKey(0)
    #cv2.destroyWindow("cutout")



last_x = -1
last_y = -1

def draw_circle(event,x,y,flags,param):
    global last_x
    global last_y
    if event == cv2.EVENT_LBUTTONDOWN:
        if (last_x == -1):
            last_x = x
            last_y = y
        else:
            calculate_reward(img, (min(last_x, x), min(last_y, y), abs(last_x-x), abs(last_y-y)))
            cv2.rectangle(img, (last_x, last_y), (x, y), 100, -1)
            last_x = -1
            last_y = -1
    




if __name__ == "__main__":
    img = cv2.imread("images/python.png")
    img = cv2.Canny(img, 50, 150)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5,5)))
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bbs = [cv2.boundingRect(c) for c in contours]
    innerMost = [c for c in bbs if all([has_no_child(c, x) for x in bbs])]
    for bb in innerMost:
        x, y, w, h = bb
        #cv2.rectangle(img, (x, y), (x+w, y+h), 127, 3)


    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)
    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break