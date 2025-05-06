# This file contains classes for all three detectors used in Chapter 5

# The main function can be used to compare predictions and speeds of all
# three detectors on the same images

from abc import ABC, abstractmethod
from time import perf_counter_ns
from cv2.typing import MatLike
import cv2
from bounding_box import BoundingBox, BoundingBoxType, RectI
from ultralytics import YOLO
from ultralytics.engine.results import Results
from utils.get_files_in_folder import get_files
from wed.cv.detector import find_elements_cv
from stable_baselines3 import PPO
import numpy as np
from numpy.typing import NDArray
from typing import cast
from wed.rl.envs.common import Action
from wed.utils.draw_bb import draw_bounding_boxes

class Detector(ABC):
    """Class providing a unified API for detectors"""
    @abstractmethod
    def predict(self, img: MatLike) -> list[BoundingBox]:
        """Locates elements in the provided image.

        Args:
            img (MatLike): The image to be analyzed (use `cv2.imread` to open the image).

        Returns:
            list[BoundingBox]: List of bounding box predictions
        """
        ...

    def predict_timed(self, img: MatLike, verbose: bool = True) -> tuple[list[BoundingBox], float]:
        """Locates elements in the provided image and measures the time taken.

        Args:
            img (MatLike): The image to be analyzed (use `cv2.imread` to open the image).
            verbose (bool, optional): Whether the time should be printed to stdout. Defaults to True.

        Returns:
            tuple[list[BoundingBox], float]: List of bounding box predictions. The time in nanoseconds.
        """
        start = perf_counter_ns()
        result = self.predict(img)
        end = perf_counter_ns()
        if verbose:
            print(f"Prediction from {self.__class__.__name__} of {len(result)} boxes took {((end-start)/1e6):.2f} ms.")
        return result, end-start

class YoloDetector(Detector):
    """YOLO based detector"""
    def __init__(self, model_path: str, device: str = "auto") -> None:
        """
        Args:
            model_path (str): Path to a model file.
            device (str, optional): On what device the prediction should run. If not specified, chooses best automatically. Example: "gpu0", "cpu". Defaults to "auto".
        """
        super().__init__()
        self.model = YOLO(model_path, verbose=False)
        self.device = device

    def predict(self, img: MatLike) -> list[BoundingBox]:
        results: list[Results] = self.model(img, verbose=False, device=self.device)
        if (len(result) < 0 or results[0].boxes is None):
            return []
        return [BoundingBox(b.tolist(), BoundingBoxType.CENTER) for b in results[0].boxes.xywhn]
    
class CVDetector(Detector):
    def __init__(self) -> None:
        super().__init__()

    def predict(self, img: MatLike) -> list[BoundingBox]:
        bbs, _ = find_elements_cv(img, True)
        return bbs

class RLDetector(Detector):
    def __init__(self, model_path: str, device: str = "auto") -> None:
        """
        Args:
            model_path (str): Path to a model file.
            device (str, optional): On what device the prediction should run. If not specified, chooses best automatically. Example: "gpu0", "cpu". Defaults to "auto".
        """
        super().__init__()
        self.model = PPO.load(model_path, device=device)
        self.height = 84
        self.width = 84
        self.view: list[float] = [0, 0, 1, 1]
        self.device = device

    def _scaling(self, img: MatLike, width, height):
        """Scales image to corret size using correct interpolation methods"""
        h, w = img.shape[:2]
        if width > w: # Upscale
            interp_x = cv2.INTER_NEAREST
        else: # Downscale
            interp_x = cv2.INTER_AREA

        img_x_scaled = cast(NDArray[np.uint8], cv2.resize(img, (width, h), interpolation=interp_x))

        if height > h: # Upscale
            interp_y = cv2.INTER_NEAREST
        else: # Downscale
            interp_y = cv2.INTER_AREA

        img_final = cast(NDArray[np.uint8], cv2.resize(img_x_scaled, (width, height), interpolation=interp_y))
        return img_final

    def _get_cutout_coords(self) -> RectI:
        """Returns coordinates of the cutout window ensuring at least 1x1 image"""
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
        """Gets the current observation"""
        x1, x2, y1, y2 = self._get_cutout_coords()

        view_cutout = self.dilated[y1:y2, x1:x2]
        view_scaled = self._scaling(view_cutout, self.width, self.height)

        img = np.expand_dims(view_scaled, axis=0)
        
        return {"image":img, "view": np.array(self.view, dtype=np.float32)}
    
    def _step(self, action: Action) -> tuple[bool, BoundingBox | None]:
        """Performs one step of the env based on the provided action"""
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

        model_state = None # Only used for recurrent policies (not used in the thesis)
        while not terminated and steps < 10000:
            obs = self._get_observation()
            action, model_state = self.model.predict(obs, model_state, deterministic=True)
            if (box_steps > 1000):
                action = Action.STOP
            terminated, bb = self._step(Action(action))
            terminated = terminated or not np.any(self.dilated)
            if bb is not None:
                box_steps = 0
                x1, x2, y1, y2 = self._get_cutout_coords()
                labels_to_remove = np.unique(labels[y1:y2, x1:x2])
                labels_to_remove = labels_to_remove[labels_to_remove != 0]
                self.dilated[np.isin(labels, labels_to_remove)] = 0
                predictions.append(bb)
                self.view = [0, 0, 1, 1]

            steps += 1
            box_steps += 1

        return predictions

if __name__ == "__main__":
    cv_detector = CVDetector()
    yolo_detector = YoloDetector(r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\wed\runs\detect\train2\weights\best.pt")
    rl_detector = RLDetector(r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\wed\rl\logs\20250502-200233\best_model\best_model.zip")

    images = r"yolo\dataset\images\test"
    #paths = get_files(images, shuffle=True, seed=0)
    paths = [r"C:\Users\Jindra\Downloads\gov.jpg", r"C:\Users\Jindra\Downloads\isik.jpg"]*20
    for i in paths:
        image = cv2.imread(i)
        cv_img, yolo_img, rl_img = [image.copy() for _ in range(3)]

        result, _ = cv_detector.predict_timed(image)
        draw_bounding_boxes(cv_img, result, (0, 0, 255))

        result, _ = yolo_detector.predict_timed(image)
        draw_bounding_boxes(yolo_img, result, (0, 0, 255))

        #result, _ = rl_detector.predict_timed(image)
        #draw_bounding_boxes(rl_img, result, (0, 0, 255))

        cv2.imshow("CV detector", cv_img)
        cv2.imshow("YOLO detector", yolo_img)
        cv2.imshow("RL detector", rl_img)
        if cv2.waitKey(0) == ord('q'):
            break