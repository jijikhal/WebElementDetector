import cv2
import random
from os import listdir
from os.path import isfile, join

def nothing(x):
    pass

def load_image(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #return cv2.bilateralFilter(gray, 21, 100, 100)
    return gray

# Create a black image, a window
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('kernel','image',1,50,nothing)
cv2.createTrackbar('s1','image',0,255,nothing)
cv2.createTrackbar('s2','image',0,255,nothing)

random.seed(1)
folder = "dataset_big"
files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
img = load_image(random.choice(files))
to_show = img.copy()

while(1):
    cv2.imshow('image',to_show)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k == ord(" "):
        img = load_image(random.choice(files))

    # get current positions of four trackbars
    r = cv2.getTrackbarPos('kernel','image')
    g = cv2.getTrackbarPos('s1','image')
    b = cv2.getTrackbarPos('s2','image')

    to_show = cv2.bilateralFilter(img, r, g, b)
    #to_show = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, r, g)

cv2.destroyAllWindows()