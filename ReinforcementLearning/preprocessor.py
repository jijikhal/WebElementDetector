from cv2.typing import MatLike
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import random


def process(img: MatLike) -> MatLike:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.bilateralFilter(gray, 21, 50, 0)
    #binary = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 3.5)
    binary = cv2.Canny(smoothed, 22, 22, L2gradient=True)
    big_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    small_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    big_count, _, big_stats, _ = cv2.connectedComponentsWithStats(big_close, connectivity=8)
    small_count, _, small_stats, _ = cv2.connectedComponentsWithStats(small_close, connectivity=8)

    """for i in range(1, small_count):
        x, y, w, h, area = small_stats[i]
        if (min(w,h) < 10 or max(w, h) < 20):
            continue
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)"""

    for i in range(1, big_count):
        x, y, w, h, area = big_stats[i]
        if (min(w,h) < 10 or max(w, h) < 20):
            continue
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #cv2.imshow("gray", gray)
    #cv2.imshow("smooth", smoothed)
    #cv2.imshow("edges", binary)
    #cv2.imshow("big", big_close)
    #cv2.imshow("small", small_close)
    cv2.imshow("img", img)

    input = cv2.resize(big_close, (100, 100), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("smaller", cv2.resize(input, (1440, 900), interpolation=cv2.INTER_NEAREST))
    #cv2.imshow("smaller", input)
    return big_close

if __name__ == "__main__":
    random.seed(69)
    folder = "dataset_big"
    files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    while True:
        img = cv2.imread(random.choice(files))
        process(img)
        if cv2.waitKey(0) & 0xFF == 27:
            break
    cv2.destroyAllWindows()