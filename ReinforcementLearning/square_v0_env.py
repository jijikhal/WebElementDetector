import gymnasium
from gymnasium import spaces
from gymnasium.envs.registration import register
from bounding_box import BoundingBox, RectF, RectI, BoundingBoxType
import numpy as np
import cv2
from cv2.typing import MatLike
import math
from stable_baselines3.common.env_checker import check_env
from gymnasium.wrappers.rescale_action import RescaleAction

register(
    id='square-v0',
    entry_point='square_v0_env:SquareEnv'
)

class SquareEnv(gymnasium.Env):
    metadata = {'render_modes': ['human'], 'render_fps':1} 
    def __init__(self, height: int = 100, width: int = 100, render_mode=None) -> None:
        super().__init__()
        self.height: int = height
        self.width: int = width
        self.render_mode = render_mode
        self.steps: int = 0
        self.img: MatLike
        self.bb: BoundingBox

        self.action_space = spaces.Box(low=0, high=np.array([1.0, 1.0, 1.0, 1.0]), shape=(4,), dtype=np.float32)

        self.observation_space = spaces.Box(low=0, high=255, shape=(1, height, width), dtype=np.uint8)

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        self.steps = 0

        self.img = np.zeros((self.height, self.width), np.uint8)
        self.bb = BoundingBox((0.5, 0.5, 0.1, 0.1), BoundingBoxType.CENTER)
        x, y, w, h = self.bb.get_rect(self.height, self.width)
        self.img[y:y+h, x:x+w] = 255
        # Convert to channel-first format. See: https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
        self.img = np.expand_dims(self.img, axis=0)

        obs = self.img
        info = {}

        if (self.render_mode == 'human'):
            self.render()

        return obs, info
    
    def calculate_reward(self, rect: RectF) -> tuple[float, bool]:
        bb = BoundingBox(rect, BoundingBoxType.CENTER)
        x1, y1, _, _= bb.get_bb_middle()
        x2, y2, _, _ = self.bb.get_bb_middle()

        total_reward = self.bb.intersection_over_union(bb)
        if (total_reward <= 0):
            total_reward -= math.sqrt((x1-x2)**2+(y1-y2)**2)

        return total_reward, False

    def step(self, action):
        self.steps += 1
        x, y, w, h = float(action[0]), float(action[1]), float(action[2]), float(action[3])
        bb = BoundingBox((x, y, w, h), BoundingBoxType.CENTER)
        reward, terminated = self.calculate_reward((x, y, w, h))
        stoped = self.steps >= 1
        x, y, w, h = bb.get_rect(self.height, self.width)

        # Uncomment the following line to see the guess
        #self.img[0][y:y+h, x:x+w] = 120

        obs = self.img
        info = {}

        if (self.render_mode == 'human'):
            self.render()

        return obs, reward, terminated, stoped, info
    
    def render(self):
        cv2.imshow("square-v0 render", self.img[0])
        cv2.waitKey(10)


if __name__ == "__main__":
    env = gymnasium.make('square-v0', render_mode='human')
    env = RescaleAction(env, -1, 1)

    print("check begin")
    check_env(env)
    print("check end")

    obs = env.reset()[0]

    for i in range(10):
        rand_action = env.action_space.sample()
        print(rand_action)
        obs, reward, terminated, _, _ = env.step(rand_action)
        print(reward, terminated)