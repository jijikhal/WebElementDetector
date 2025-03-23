import random
import gymnasium
from gymnasium import spaces
from gymnasium.envs.registration import register
from sympy import use
from bounding_box import BoundingBox, RectF, RectI, BoundingBoxType
import numpy as np
import cv2
from cv2.typing import MatLike
from stable_baselines3.common.env_checker import check_env
import bounding_box
from os import listdir
from os.path import isfile, join
from time import time

register(
    id='square-v8-discrete',
    entry_point='square_v8_env_discrete:SquareEnv'
)

SHRINK_LEFT = 0
SHRINK_RIGHT = 1
SHRINK_TOP = 2
SHRINK_BOTTOM = 3
SHRINK_LEFT_SMALL = 4
SHRINK_RIGHT_SMALL = 5
SHRINK_TOP_SMALL = 6
SHRINK_BOTTOM_SMALL = 7
STOP = 8

def draw_rect(bb: RectI, image: MatLike) -> None:
    x, y, w, h = bb
    cv2.rectangle(image, (x, y), (x+w-1, y+h-1), (255,), 1)

def find_bounding_boxes(img: MatLike) -> list[BoundingBox]:
    copy_for_show = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.bilateralFilter(gray, 21, 50, 0)
    binary = cv2.Canny(smoothed, 22, 22, L2gradient=True)
    big_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    big_count, _, big_stats, _ = cv2.connectedComponentsWithStats(big_close, connectivity=8)
    img_h, img_w, _ = img.shape

    result: list[BoundingBox] = []

    for i in range(0, big_count):
        x, y, w, h, area = big_stats[i]
        if (min(w,h) < 10 or max(w, h) < 20):
            cv2.rectangle(copy_for_show, (x, y), (x + w + 1, y + h + 1), (0, 0, 255), 1)
            continue
        result.append(BoundingBox((x/img_w, y/img_h, (w+1)/img_w, (h+1)/img_h), BoundingBoxType.TOP_LEFT))
        cv2.rectangle(copy_for_show, (x, y), (x + w + 1, y + h + 1), (0, 255, 0), 2)

    merged: list[BoundingBox] = []
    used = set()
    result.sort(key=lambda x: x.area())
    for b in result:
        if (b in used):
            continue
        used.add(b)
        to_merge_with = [x for x in result if b.is_intersecting(x) and x not in used]
        m = b
        for t in to_merge_with:
            used.add(t)
            m = b.merge(t)
        merged.append(m)

    for b in merged:
        x1, y1, x2, y2 = b.get_bb_corners()
        cv2.rectangle(copy_for_show, (round(x1*img_w), round(y1*img_h)), (round(x2*img_w), round(y2*img_h)), (255, 0, 0), 2)

    merged.sort(key=lambda x: x.area(), reverse=True)

    #cv2.imshow("img", copy_for_show)
    #cv2.waitKey(0)
    
    return merged

class SquareEnv(gymnasium.Env):
    metadata = {'render_modes': ['human','none'], 'render_fps':1} 
    def __init__(self, height: int = 100, width: int = 100, render_mode=None, dataset_folder: str = "dataset_big") -> None:
        super().__init__()
        self.height: int = height
        self.width: int = width
        self.render_mode = render_mode
        self.steps: int = 0
        self.image_paths = [join(dataset_folder, f) for f in listdir(dataset_folder) if isfile(join(dataset_folder, f))]
        self.base_img = cv2.imread(random.choice(self.image_paths))
        self.ground_truth_labels: list[BoundingBox] = []
        self.trash: list[BoundingBox] = []
        self.view: list[float] = [0.0, 0.0, 1.0, 1.0]
        self.last_reward = 0
        self.preprocessed: MatLike | None = None

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, height, width), dtype=np.uint8)

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        random.seed(seed)
        np.random.seed(seed)

        path = random.choice(self.image_paths)
        self.base_img = cv2.imread(path)
        self.ground_truth_labels = find_bounding_boxes(self.base_img)
        self.steps = len(self.ground_truth_labels)*30
        self.view = [0, 0, 1, 1]
        self.last_reward = 0
        self.guesses = []
        self.preprocessed = None

        obs = self.get_observation()
        info = {}

        if (self.render_mode == 'human'):
            self.render()

        return obs, info
    
    def get_observation(self) -> MatLike:
        if self.preprocessed is None:
            img_h, img_w, _ = self.base_img.shape
            img = np.zeros((img_h, img_w), dtype=np.uint8)
            for l in self.ground_truth_labels:
                x1, y1, x2, y2 = l.get_bb_corners()
                cv2.rectangle(img, (round(x1*img_w), round(y1*img_h)), (round(x2*img_w)-1, round(y2*img_h)-1), (255,), 1)
            self.preprocessed = img

        #cv2.imshow("lol", self.preprocessed)

        closing = self.preprocessed
        img_h, img_w, _ = self.base_img.shape

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
        view_cutout = closing[y1:y2, x1:x2]
        view_scaled = cv2.resize(view_cutout, (self.width, self.height), interpolation=cv2.INTER_AREA)
        view_scaled[view_scaled > 0] = 255

        # Convert to channel-first format. See: https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
        return np.expand_dims(view_scaled, axis=0)
    
    def calculate_reward_dense(self, rect: RectF, stop: bool) -> tuple[float, bool]:

        guess = BoundingBox(rect, BoundingBoxType.CENTER)

        bbs = self.ground_truth_labels.copy()
        bbs.sort(key=lambda x: x.iou(guess), reverse=True)

        best_bb = bbs[0]
        max_iou = best_bb.iou(guess) if len(self.ground_truth_labels) == 1 else bbs[1].iou(guess)

        best_is_root = best_bb is self.ground_truth_labels[0]

        if stop:
            self.ground_truth_labels.remove(best_bb)
            x1, y1, x2, y2 = best_bb.get_bb_corners()
            img_h, img_w, _ = self.base_img.shape
            self.preprocessed[round(y1*img_h):round(y2*img_h), round(x1*img_w):round(x2*img_w)] = 0
            return max_iou*3, best_is_root
            
        diff = max_iou - self.last_reward
        self.last_reward = max_iou 

        return diff, False

    def step(self, action):
        self.steps -= 1
        width = self.view[2]-self.view[0]
        heigth = self.view[3]-self.view[1]
        if action == SHRINK_LEFT:
            self.view[0] = self.view[0] + width*0.15
        elif action == SHRINK_TOP:
            self.view[1] = self.view[1] + heigth*0.15
        elif action == SHRINK_RIGHT:
            self.view[2] = self.view[2] - width*0.15
        elif action == SHRINK_BOTTOM:
            self.view[3] = self.view[3] - heigth*0.15
        elif action == SHRINK_LEFT_SMALL:
            self.view[0] = self.view[0] + width*0.025
        elif action == SHRINK_TOP_SMALL:
            self.view[1] = self.view[1] + heigth*0.025
        elif action == SHRINK_RIGHT_SMALL:
            self.view[2] = self.view[2] - width*0.025
        elif action == SHRINK_BOTTOM_SMALL:
            self.view[3] = self.view[3] - heigth*0.025
        obs = self.get_observation()

        reward, terminated = 0, False

        bb = BoundingBox(self.view, BoundingBoxType.TWO_CORNERS)

        if action == STOP:
            reward, terminated = self.calculate_reward_dense(bb.get_bb_middle(), True)
            self.view = [0, 0, 1, 1]
            self.last_reward = 0
            if (self.render_mode == 'human'):
                self.render()
            return self.get_observation(), reward, terminated, False, {}
        else:
            reward, terminated = self.calculate_reward_dense(bb.get_bb_middle(), False)


        if (self.render_mode == 'human'):
            self.render()

        return obs, reward, terminated, self.steps <= 0, {}
    
    def render(self):
        if self.render_mode == 'human':
            view = self.get_observation()[0]
            show = cv2.resize(view, (500, 500), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("square-v5-discrete render", show)
            cv2.waitKey(0)
        else:
            return self.view

if __name__ == "__main__":
    env = gymnasium.make('square-v8-discrete', render_mode='none')

    print("check begin")
    check_env(env)
    print("check end")

    """total = 0
    for _ in range(10000):
        env.reset()
        #print(env.env.env.__dict__)
        total += env.env.env.steps

    print(total/10000)"""

    obs = env.reset(seed=1)[0]

    for i in range(1000):
        rand_action = env.action_space.sample()
        print(rand_action)
        obs, reward, terminated, _, _ = env.step(rand_action)
        print(reward, terminated)
        if (terminated):
            break