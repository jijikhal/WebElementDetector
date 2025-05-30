# This file contains the environment used in Section 4.8 containing multiple randomly placed shapes
import gymnasium
from gymnasium import spaces
from gymnasium.envs.registration import register
from wed.utils.bounding_box import BoundingBox, RectF, RectI, BoundingBoxType
import numpy as np
import cv2
from cv2.typing import MatLike
import math
from random import uniform
from stable_baselines3.common.env_checker import check_env
from gymnasium.wrappers import RescaleAction

register(
    id='square-v4',
    entry_point='wed.rl.envs.square_v4_env:SquareEnv'
)


def draw_triangle(bb: RectI, image: MatLike) -> None:
    x, y, w, h = bb
    side = np.random.randint(0, 4)
    if side == 0:
        points = np.array([(x+w//2, y), (x+w-1, y+h-1), (x, y+h-1)])
    elif side == 1:
        points = np.array([(x+w-1, y+h//2), (x, y+h-1), (x, y)])
    elif side == 2:
        points = np.array([(x+w//2, y+h-1), (x+w-1, y), (x, y)])
    else:
        points = np.array([(x, y+h//2), (x+w-1, y+h-1), (x+w-1, y)])
    cv2.drawContours(image, [points], 0, (255,), -1)


def draw_rect(bb: RectI, image: MatLike) -> None:
    x, y, w, h = bb
    image[y:y+h, x:x+w] = 255


def draw_ellipse(bb: RectI, image: MatLike) -> None:
    x, y, w, h = bb
    cv2.ellipse(image, (x+w//2, y+h//2), (w//2-1, h//2-1), 0, 0, 360, (255,), -1)


class SquareEnv(gymnasium.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 1}

    def __init__(self, height: int = 100, width: int = 100, render_mode=None) -> None:
        super().__init__()
        self.height: int = height
        self.width: int = width
        self.render_mode = render_mode
        self.steps: int = 0
        self.img: MatLike
        self.bb: list[BoundingBox] = []

        self.action_space = spaces.Box(low=0, high=np.array([1.0, 1.0, 1.0, 1.0]), shape=(4,), dtype=np.float32)

        self.observation_space = spaces.Box(low=0, high=255, shape=(1, height, width), dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.bb = []
        self.img = np.zeros((self.height, self.width), np.uint8)

        for _ in range(5):
            xr, yr = uniform(0.1, 0.9), uniform(0.1, 0.9)
            wr, hr = uniform(0.1, min(xr, 1-xr)), uniform(0.1, min(yr, 1-yr))
            new_bb = BoundingBox((xr, yr, wr, hr), BoundingBoxType.CENTER)
            if (not any([x.has_overlap(new_bb) for x in self.bb])):
                self.bb.append(new_bb)
                bb = new_bb.get_rect(self.width, self.height)
                shape = np.random.randint(0, 3)
                if (shape == 0):
                    draw_rect(bb, self.img)
                elif (shape == 1):
                    draw_ellipse(bb, self.img)
                else:
                    draw_triangle(bb, self.img)

        self.steps = len(self.bb)

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

        total_reward = max([x.iou(bb) for x in intersecting])

        for i in intersecting:
            x, y, w, h = i.get_rect(self.width, self.height)
            self.img[0][y:y+h, x:x+w] = 0
            self.bb.remove(i)

        return total_reward, len(self.bb) == 0

    def step(self, action):
        self.steps -= 1
        x, y, w, h = float(action[0]), float(action[1]), float(action[2]), float(action[3])
        bb = BoundingBox((x, y, w, h), BoundingBoxType.CENTER)
        reward, terminated = self.calculate_reward((x, y, w, h))
        stoped = self.steps <= 0
        x, y, w, h = bb.get_rect(self.width, self.height)

        # Uncomment the following line to see the guess
        self.img[0][y:y+h, x:x+w] = 120

        obs = self.img
        info = {}

        if (self.render_mode == 'human'):
            print("boxes:", len(self.bb))
            self.render()

        self.img[0][y:y+h, x:x+w] = 0

        return obs, reward, terminated, stoped, info

    def render(self):
        cv2.imshow("square-v4 render", self.img[0])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()


if __name__ == "__main__":
    env = gymnasium.make('square-v4', render_mode='human')
    env = RescaleAction(env, -1, 1)

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
