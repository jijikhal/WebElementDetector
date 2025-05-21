# This file contains the environment used in Section 4.13 containing one randomly placed rectangle with discrete actions
import gymnasium
from gymnasium import spaces
from gymnasium.envs.registration import register
from wed.rl.envs.common import ActionSmall
from wed.utils.bounding_box import BoundingBox, RectF, RectI, BoundingBoxType
import numpy as np
import cv2
from cv2.typing import MatLike
import math
from random import uniform
from stable_baselines3.common.env_checker import check_env

register(
    id='square-v2-discrete',
    entry_point='wed.rl.envs.square_v2_env_discrete:SquareEnv'
)


class SquareEnv(gymnasium.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 1}

    def __init__(self, height: int = 100, width: int = 100, render_mode=None) -> None:
        super().__init__()
        self.height: int = height
        self.width: int = width
        self.render_mode = render_mode
        self.steps: int = 0
        self.img: MatLike
        self.bb: BoundingBox
        self.view: list[float] = [0.0, 0.0, 1.0, 1.0]

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(low=0, high=255, shape=(1, height, width), dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        self.img = np.zeros((self.height, self.width), np.uint8)
        xr, yr = uniform(0.1, 0.9), uniform(0.1, 0.9)
        wr, hr = uniform(0.1, min(xr, 1-xr)), uniform(0.1, min(yr, 1-yr))
        self.bb = BoundingBox((xr, yr, wr, hr), BoundingBoxType.CENTER)
        x, y, w, h = self.bb.get_rect(self.width, self.height)
        self.img[y:y+h, x:x+w] = 255
        # Convert to channel-first format. See: https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
        self.img = np.expand_dims(self.img, axis=0)

        obs = self.img
        info = {}

        self.view = [0, 0, 1, 1]

        if (self.render_mode == 'human'):
            self.render()

        return obs, info

    def calculate_reward(self, rect: RectF) -> float:
        bb = BoundingBox(rect, BoundingBoxType.CENTER)
        x1, y1, _, _ = bb.get_bb_middle()
        x2, y2, _, _ = self.bb.get_bb_middle()

        total_reward = self.bb.iou(bb)
        if (total_reward <= 0):
            total_reward -= math.sqrt((x1-x2)**2+(y1-y2)**2)

        return total_reward

    def step(self, action: ActionSmall):
        self.steps += 1
        width = self.view[2]-self.view[0]
        heigth = self.view[3]-self.view[1]
        if action == ActionSmall.SHRINK_LEFT:
            self.view[0] = self.view[0] + width*0.1
        elif action == ActionSmall.SHRINK_TOP:
            self.view[1] = self.view[1] + heigth*0.1
        elif action == ActionSmall.SHRINK_RIGHT:
            self.view[2] = self.view[2] - width*0.1
        elif action == ActionSmall.SHRINK_BOTTOM:
            self.view[3] = self.view[3] - heigth*0.1

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
                x2 += 2
        if (y1 == y2):
            if (y2 == self.height):
                y1 -= 2
            elif (y1 == 0):
                y2 += 2
            else:
                y2 += 2

        view = self.img[0][y1:y2, x1:x2]
        obs = np.expand_dims(cv2.resize(view, (self.width, self.height), interpolation=cv2.INTER_NEAREST), axis=0)

        if action == ActionSmall.STOP:
            bb = BoundingBox(self.view, BoundingBoxType.TWO_CORNERS)
            reward = self.calculate_reward(bb.get_bb_middle())
            return obs, reward, True, False, {}

        if (self.render_mode == 'human'):
            self.render()

        return obs, 0, False, self.steps >= 100, {}

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
                x2 += 2

        if (y1 == y2):
            if (y2 == self.height):
                y1 -= 2
            elif (y1 == 0):
                y2 += 2
            else:
                y2 += 2
        view = self.img[0][y1:y2, x1:x2]
        show = cv2.resize(view, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("square-v2 render", show)
        cv2.waitKey(1)

    def close(self):
        if self.render_mode == 'human':
            cv2.destroyAllWindows()


if __name__ == "__main__":
    env = gymnasium.make('square-v2-discrete', render_mode='human')

    print("check begin")
    check_env(env)
    print("check end")

    for i in range(10):
        terminated, truncated = False, False
        obs = env.reset()[0]
        while (not terminated and not truncated):
            rand_action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(rand_action)
            print(reward, terminated, rand_action)
            if cv2.waitKey(0) == ord('q'):
                env.close()
                exit(0)
