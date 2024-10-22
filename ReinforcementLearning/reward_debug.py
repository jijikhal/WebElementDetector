import numpy as np
import cv2

def is_fully_inside(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    return x1 <= x2 and y1 <= y2 and x1+w1 >= x2+w2 and y1+h1 >= y2+h2

def has_no_child(rect1, rect2):
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
            cv2.rectangle(img, (last_x, last_y), (x, y), 255, 1)
            last_x = -1
            last_y = -1
    




if __name__ == "__main__":
    img = cv2.imread("images/python.jpg")
    img = cv2.Canny(img, 50, 150)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5,5)))
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bbs = [cv2.boundingRect(c) for c in contours]
    innerMost = [c for c in bbs if all([has_no_child(c, x) for x in bbs])]
    #innerMost = [c for c, h in zip(contours, hierarchy[0]) if h[2] == -1]
    for bb in innerMost:
        x, y, w, h = bb
        cv2.rectangle(img, (x, y), (x+w, y+h), 127, 1)


    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)
    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break