import random
import gymnasium
from gymnasium import spaces
from gymnasium.envs.registration import register
from sklearn.utils import resample
from bounding_box import BoundingBox, RectF, RectI, BoundingBoxType
import numpy as np
import cv2
from cv2.typing import MatLike
from random import uniform
from stable_baselines3.common.env_checker import check_env
from gymnasium.wrappers.rescale_action import RescaleAction

register(
    id='square-v6',
    entry_point='square_v6_env:SquareEnv'
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
    cv2.drawContours(image, [points], 0, (255,), 1)

def draw_rect(bb: RectI, image: MatLike) -> None:
    x, y, w, h = bb
    cv2.rectangle(image, (x, y), (x+w-1, y+h-1), (255,), 1)

def draw_ellipse(bb: RectI, image: MatLike) -> None:
    x, y, w, h = bb
    cv2.ellipse(image, (x+w//2, y+h//2), (w//2-1, h//2-1), 0, 0, 360, (255,), 1)


class SquareEnv(gymnasium.Env):
    metadata = {'render_modes': ['human'], 'render_fps':1} 
    def __init__(self, height: int = 200, width: int = 200, render_mode=None) -> None:
        super().__init__()
        self.height: int = height
        self.width: int = width
        self.render_mode = render_mode
        self.steps: int = 0
        self.img: MatLike
        self.bb: list[BoundingBox] = []

        self.action_space = spaces.Box(low=0, high=np.array([1.0, 1.0, 1.0, 1.0]), shape=(4,), dtype=np.float32)

        self.observation_space = spaces.Box(low=0, high=255, shape=(1, height, width), dtype=np.uint8)

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        random.seed(seed)
        np.random.seed(seed)
        self.bb = []
        self.img = np.zeros((self.height, self.width), np.uint8)

        for _ in range(25):
            min_size = 0.05
            xr, yr = uniform(min_size, 1-min_size), uniform(min_size, 1-min_size)
            wr, hr = uniform(min_size, min(xr, 1-xr)), uniform(min_size, min(yr, 1-yr))
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
        elif (len(intersecting) == 1):
            total_reward = intersecting[0].intersection_over_union(bb)
        else:
            intersecting.sort(key=lambda x: x.intersection_over_union(bb), reverse=True)
            best_score = intersecting[0].intersection_over_union(bb)
            other_overlap = sum([x.overlap(bb) for x in intersecting[1:]])
            total_reward = best_score - 1*(other_overlap/bb.area())

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
            #print("boxes:", len(self.bb))
            self.render()

        self.img[0][y:y+h, x:x+w] = 0

        return obs, reward, terminated, stoped, info
    
    def render(self):
        #return self.img[0]
        cv2.imshow("square-v6 render", self.img[0])
        cv2.waitKey(0)


if __name__ == "__main__":
    env = gymnasium.make('square-v6', render_mode='human')
    # env = RescaleAction(env, -1, 1)

    print("check begin")
    #check_env(env)
    print("check end")

    """total = 0
    for _ in range(10000):
        env.reset()
        total += env.steps

    print(total/10000)"""

    obs = env.reset()[0]

    for i in range(10):
        rand_action = env.action_space.sample()
        print(rand_action)
        obs, reward, terminated, _, _ = env.step(rand_action)
        print(reward, terminated)
        if (terminated):
            break