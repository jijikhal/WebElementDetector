import gymnasium
from gymnasium import spaces
from gymnasium.envs.registration import register
from wed.bounding_box import BoundingBox, RectF, RectI, BoundingBoxType
import numpy as np
import cv2
from cv2.typing import MatLike
import math
from random import uniform
from stable_baselines3.common.env_checker import check_env
from gymnasium.wrappers import RescaleAction

register(
    id='square-v5-discrete',
    entry_point='wed.rl.envs.square_v5_env_discrete:SquareEnv'
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

class SquareEnv(gymnasium.Env):
    metadata = {'render_modes': ['human'], 'render_fps':1} 
    def __init__(self, height: int = 100, width: int = 100, render_mode=None) -> None:
        super().__init__()
        self.height: int = height
        self.width: int = width
        self.render_mode = render_mode
        self.steps: int = 0
        self.img: MatLike
        self.bb: list[BoundingBox] = []
        self.view: list[float] = [0.0, 0.0, 1.0, 1.0]

        self.action_space = spaces.Discrete(9)

        self.observation_space = spaces.Box(low=0, high=255, shape=(1, height, width), dtype=np.uint8)

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        self.bb = []
        self.img = np.zeros((self.height, self.width), np.uint8)

        for _ in range(5):
            xr, yr = uniform(0.1, 0.9), uniform(0.1, 0.9)
            wr, hr = uniform(0.1, min(xr, 1-xr)), uniform(0.1, min(yr, 1-yr))
            new_bb = BoundingBox((xr, yr, wr, hr), BoundingBoxType.CENTER)
            if (not any([x.has_overlap(new_bb) for x in self.bb])):
                self.bb.append(new_bb)
                x, y, w, h = new_bb.get_rect(self.width, self.height)
                cv2.rectangle(self.img, (x,y,w,h), (255,), 1)

        self.steps = len(self.bb)*30
        self.view = [0, 0, 1, 1]
        
        # Convert to channel-first format. See: https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
        self.img = np.expand_dims(self.img, axis=0)

        obs = self.img
        info = {}

        if (self.render_mode == 'human'):
            self.render()

        return obs, info
    
    def calculate_reward(self, rect: RectF) -> tuple[float, bool]:

        bb = BoundingBox(rect, BoundingBoxType.CENTER)

        intersecting = [x for x in self.bb if x.has_overlap(bb)]

        if (len(intersecting) == 0):
                return -min([x.get_distance(bb) for x in self.bb]), len(self.bb) == 0
        elif (len(intersecting) == 1):
            total_reward = intersecting[0].iou(bb)
        else:
            intersecting.sort(key=lambda x: x.iou(bb), reverse=True)
            best_score = intersecting[0].iou(bb)
            other_overlap = sum([x.overlap(bb) for x in intersecting[1:]])
            total_reward = best_score - 1*(other_overlap/bb.area())

        for i in intersecting:
            x, y, w, h = i.get_rect(self.width, self.height)
            self.img[0][y:y+h, x:x+w] = 0
            self.bb.remove(i)

        return total_reward, len(self.bb) == 0

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

        # Calculate area so that it is always at least 1x1 pixels in a valid spot
        x1 = int(self.view[0]*self.width)
        y1 = int(self.view[1]*self.height)
        x2 = max(int(self.view[2]*self.width), x1)
        y2 = max(int(self.view[3]*self.height), y1)
        if (x1 == x2):
            if (x2 == self.width):
                x1 -= 2
            elif (x1 == 0):
                x2 += 2
            else:
                x2 +=2
        if (y1 == y2):
            if (y2 == self.height):
                y1 -= 2
            elif (y1 == 0):
                y2 += 2
            else:
                y2 +=2

        view = self.img[0][y1:y2, x1:x2]
        obs = np.expand_dims(cv2.resize(view, (self.width, self.height), interpolation=cv2.INTER_NEAREST), axis=0)

        if action == STOP:
            bb = BoundingBox(self.view, BoundingBoxType.TWO_CORNERS)
            reward, terminated = self.calculate_reward(bb.get_bb_middle())
            self.view = [0, 0, 1, 1]
            return self.img, reward, terminated, False, {}

        if (self.render_mode == 'human'):
            self.render()

        return obs, 0, False, self.steps <= 0, {}
    
    def render(self):
        x1 = int(self.view[0]*self.width)
        y1 = int(self.view[1]*self.height)
        x2 = max(int(self.view[2]*self.width), x1)
        y2 = max(int(self.view[3]*self.height), y1)

        if (x1 == x2):
            if (x2 == self.width):
                x1 -= 2
            elif (x1 == 0):
                x2 += 2
            else:
                x2 +=2

        if (y1 == y2):
            if (y2 == self.height):
                y1 -= 2
            elif (y1 == 0):
                y2 += 2
            else:
                y2 +=2
        view = self.img[0][y1:y2, x1:x2]
        show = cv2.resize(view, (500, 500), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("square-v5 render", show)
        cv2.waitKey(0)


if __name__ == "__main__":
    env = gymnasium.make('square-v5-discrete', render_mode='human')

    print("check begin")
    #check_env(env)
    print("check end")

    obs = env.reset()[0]

    for i in range(100):
        rand_action = env.action_space.sample()
        print(rand_action)
        obs, reward, terminated, _, _ = env.step(rand_action)
        print(reward, terminated)
        if (terminated):
            break