from cv2.typing import MatLike
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import random


def process(mat: MatLike) -> MatLike:
    mat = cv2.Canny(mat, 100, 200)
    #mat = cv2.morphologyEx(mat, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return mat

if __name__ == "__main__":
    random.seed(1)
    folder = "dataset"
    files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    while True:
        img = cv2.imread(random.choice(files))
        processed = cv2.cvtColor(process(img), cv2.COLOR_GRAY2BGR)
        cv2.imshow("lol", np.concatenate([img, processed], 1))
        if cv2.waitKey(0) & 0xFF == 27:
            break
    cv2.destroyAllWindows()