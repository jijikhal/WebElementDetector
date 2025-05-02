from abc import ABC, abstractmethod
from time import perf_counter_ns
from cv2.typing import MatLike
import cv2
from bounding_box import BoundingBox, BoundingBoxType
from ultralytics import YOLO
from ultralytics.engine.results import Results
from wed.cv.detector import find_elements_cv
from stable_baselines3 import PPO
import numpy as np
from numpy.typing import NDArray
from typing import cast
from wed.rl.envs.common import Action
from os.path import join, isfile
from os import listdir
from wed.utils.draw_bb import draw_bounding_boxes

class Detector(ABC):
    @abstractmethod
    def predict(self, img: MatLike) -> list[BoundingBox]:
        ...

    def predict_timed(self, img: MatLike, verbose: bool = True) -> tuple[list[BoundingBox], float]:
        start = perf_counter_ns()
        result = self.predict(img)
        end = perf_counter_ns()
        print(f"Prediction from {self.__class__.__name__} of {len(result)} boxes took {((end-start)/1e6):.2f} ms.")
        return result, end-start

class YoloDetector(Detector):
    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.model = YOLO(model_path, verbose=False)

    def predict(self, img: MatLike) -> list[BoundingBox]:
        results: list[Results] = self.model(img, verbose=False)#, device="cpu")
        bbs: list[BoundingBox] = [BoundingBox(b.tolist(), BoundingBoxType.CENTER) for b in results[0].boxes.xywhn]
        return bbs
    
class CVDetector(Detector):
    def __init__(self) -> None:
        super().__init__()

    def predict(self, img: MatLike) -> list[BoundingBox]:
        bbs, _ = find_elements_cv(img, True)
        return bbs

class RLDetector(Detector):
    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.model = PPO.load(model_path)
        self.height = 84
        self.width = 84
        self.view: list[float] = [0, 0, 1, 1]

    def _scaling(self, img: MatLike, width, height):
        h, w = img.shape[:2]
        if width > w:
            interp_x = cv2.INTER_NEAREST
        else:
            interp_x = cv2.INTER_AREA

        img_x_scaled = cast(NDArray[np.uint8], cv2.resize(img, (width, h), interpolation=interp_x))

        if height > h:
            interp_y = cv2.INTER_NEAREST
        else:
            interp_y = cv2.INTER_AREA

        img_final = cast(NDArray[np.uint8], cv2.resize(img_x_scaled, (width, height), interpolation=interp_y))
        return img_final

    def _get_cutout_coords(self) -> tuple[int,int,int,int]:
        img_h, img_w = self.dilated.shape

        # Calculate area so that it is always at least 1x1 pixels in a valid spot
        x1 = int(self.view[0]*img_w)
        y1 = int(self.view[1]*img_h)
        x2 = max(int(self.view[2]*img_w), x1)
        y2 = max(int(self.view[3]*img_h), y1)
        if (x1 == x2):
            if (x2 == img_w):
                x1 -= 2
            elif (x1 == 0):
                x2 += 2
            else:
                x2 +=2
        if (y1 == y2):
            if (y2 == img_h):
                y1 -= 2
            elif (y1 == 0):
                y2 += 2
            else:
                y2 +=2

        return x1, x2, y1, y2

    def _get_observation(self) -> dict[str, MatLike]:
        x1, x2, y1, y2 = self._get_cutout_coords()

        view_cutout = self.dilated[y1:y2, x1:x2]
        view_scaled = self._scaling(view_cutout, self.width, self.height)

        img = np.expand_dims(view_scaled, axis=0)
        
        return {"image":img, "view": np.array(self.view, dtype=np.float32)}
    
    def _step(self, action: Action) -> tuple[bool, BoundingBox|None]:
        width = self.view[2]-self.view[0]
        heigth = self.view[3]-self.view[1]
        if action == Action.SHRINK_LEFT:
            self.view[0] = self.view[0] + width*0.15
        elif action == Action.SHRINK_TOP:
            self.view[1] = self.view[1] + heigth*0.15
        elif action == Action.SHRINK_RIGHT:
            self.view[2] = self.view[2] - width*0.15
        elif action == Action.SHRINK_BOTTOM:
            self.view[3] = self.view[3] - heigth*0.15
        elif action == Action.SHRINK_LEFT_SMALL:
            self.view[0] = self.view[0] + width*0.025
        elif action == Action.SHRINK_TOP_SMALL:
            self.view[1] = self.view[1] + heigth*0.025
        elif action == Action.SHRINK_RIGHT_SMALL:
            self.view[2] = self.view[2] - width*0.025
        elif action == Action.SHRINK_BOTTOM_SMALL:
            self.view[3] = self.view[3] - heigth*0.025

        bb = None

        terminated = False
        if action == Action.STOP:
            bb = BoundingBox(self.view, BoundingBoxType.TWO_CORNERS)
            terminated = self.view == [0, 0, 1, 1]

        return terminated, bb

    def predict(self, img: MatLike) -> list[BoundingBox]:
        self.view: list[float] = [0, 0, 1, 1]
        img_h, img_w, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        smooth = cv2.bilateralFilter(gray, 21, 50, 10)
        canny = cv2.Canny(smooth, 100, 200, L2gradient=True)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.dilated = cv2.morphologyEx(canny, cv2.MORPH_DILATE, kernel)

        _, labels = cv2.connectedComponents(self.dilated, connectivity=8)

        terminated = False
        steps = 0
        predictions: list[BoundingBox] = []
        box_steps = 0

        start = perf_counter_ns()
        while not terminated and steps < 10000:
            print((perf_counter_ns()-start)/1e6)
            obs = self._get_observation()
            action, _ = self.model.predict(obs, deterministic=True)
            start = perf_counter_ns()
            if (box_steps > 100):
                action = Action.STOP
            terminated, bb = self._step(action)
            terminated = terminated or not np.any(self.dilated)
            if bb is not None:
                box_steps = 0
                #img_cpy = img.copy()
                #cv2.rectangle(img_cpy, bb.get_rect(img_w, img_h), (0, 0, 255), 2)
                #cv2.imshow("selection", img_cpy)
                x1, x2, y1, y2 = self._get_cutout_coords()
                #print(x1, x2, y1, y2)
                labels_to_remove = np.unique(labels[y1:y2, x1:x2])
                labels_to_remove = labels_to_remove[labels_to_remove != 0]
                #print(labels_to_remove)
                #dil_copy = self.dilated.copy()
                #dil_copy[np.isin(labels, labels_to_remove) & (self.dilated == 255)] = 128
                self.dilated[np.isin(labels, labels_to_remove)] = 0
                #cv2.imshow("after", dil_copy)
                predictions.append(bb)
                #cv2.waitKey(0)
                self.view = [0, 0, 1, 1]

            steps += 1
            box_steps += 1

        return predictions

if __name__ == "__main__":
    #cv_detector = CVDetector()
    #yolo_detector = YoloDetector(r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\wed\yolo\runs\detect\train5\weights\best.pt")
    start = perf_counter_ns()
    rl_detector = RLDetector(r"rl\logs\v9d_curriculum0.7_grayscale_squeez_large_samedata\best_model\best_model.zip")
    print(f"Init took {(perf_counter_ns()-start)/1e9} s.")

    images = r"yolo\dataset\images\test"
    #paths = [join(images, f) for f in listdir(images) if isfile(join(images, f))]
    paths = [r"C:\Users\Jindra\Downloads\isik.jpg"]
    for i in paths:
        image = cv2.imread(i)
        cv_img, yolo_img, rl_img = [image.copy() for _ in range(3)]

        #result, _ = cv_detector.predict_timed(image)
        #draw_bounding_boxes(cv_img, result, (0, 0, 255))

        #result, _ = yolo_detector.predict_timed(image)
        #draw_bounding_boxes(yolo_img, result, (0, 0, 255))

        result, _ = rl_detector.predict_timed(image)
        draw_bounding_boxes(rl_img, result, (0, 0, 255))

        cv2.imshow("CV detector", cv_img)
        cv2.imshow("YOLO detector", yolo_img)
        cv2.imshow("RL detector", rl_img)
        if cv2.waitKey(0) == ord('q'):
            break