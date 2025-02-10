import random
import gymnasium
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.wrappers import RescaleAction
from bounding_box import BoundingBox, RectF, RectI, BoundingBoxType
import numpy as np
import cv2
from cv2.typing import MatLike
from stable_baselines3.common.env_checker import check_env
from image_generator import generete_hierarchy, Node

register(
    id='square-v7',
    entry_point='square_v7_env:SquareEnv'
)

def draw_rect(bb: RectI, image: MatLike) -> None:
    x, y, w, h = bb
    cv2.rectangle(image, (x, y), (x+w-1, y+h-1), (255,), 1)

class SquareEnv(gymnasium.Env):
    metadata = {'render_modes': ['human'], 'render_fps':1} 
    def __init__(self, height: int = 100, width: int = 100, render_mode=None) -> None:
        super().__init__()
        self.height: int = height
        self.width: int = width
        self.render_mode = render_mode
        self.steps: int = 0
        self.img: MatLike
        self.bb: list[Node] = []

        self.action_space = spaces.Box(low=0, high=np.array([1.0, 1.0, 1.0, 1.0]), shape=(4,), dtype=np.float32)

        self.observation_space = spaces.Box(low=0, high=255, shape=(1, height, width), dtype=np.uint8)

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        random.seed(seed)
        np.random.seed(seed)
        self.img, self.bb = generete_hierarchy((self.height, self.width), seed)

        self.steps = len(self.bb)
        
        # Convert to channel-first format. See: https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
        self.img = np.expand_dims(self.img, axis=0)

        obs = self.img
        info = {}

        if (self.render_mode == 'human'):
            self.render()

        return obs, info
    
    def calculate_reward(self, rect: RectF) -> tuple[float, bool]:

        guess = BoundingBox(rect, BoundingBoxType.CENTER)
        end = False

        intersecting = [x for x in self.bb if x.bb.is_intersecting(guess) or guess.fully_contains(x.bb)]

        if (len(intersecting) == 0):
            # If the bounding box does not contain anything, also include objects that contain it
            intersecting = [x for x in self.bb if x.bb.fully_contains(guess)]
            intersecting.sort(key=lambda x: x.level, reverse=True)
            intersecting = [intersecting[0]]
        
        if (len(intersecting) == 1):
            hit = intersecting[0]

            ## Modification to only reward inner boxes
            if hit.level < 2:
                still_valid = list(filter(lambda x: x.level >= 2, self.bb))
                if len(still_valid) > 0:
                    return -len(still_valid), True
                return 0, True
            ## END

            total_reward = hit.bb.intersection_over_union(guess)
            remaining_children_area = 0
            for c in hit.children:
                remaining_children_area += c.bb.area()
            total_reward -= remaining_children_area / hit.bb.area()
            
            hit.remove(self.img[0], self.bb)
            end = len(self.bb) == 0

        else:
            intersecting.sort(key=lambda x: x.level, reverse=True)
            lowest_child = intersecting[0]

            ## Modification to only reward inner boxes
            if lowest_child.level < 2:
                still_valid = list(filter(lambda x: x.level >= 2, self.bb))
                if len(still_valid) > 0:
                    return -len(still_valid), True
                return 0, True
            ## END

            best_score = lowest_child.bb.intersection_over_union(guess)
            other_overlap = sum([x.bb.overlap(guess) for x in intersecting[1:]])
            total_reward = best_score
            if (guess.area() != 0):
                total_reward -= 1*(other_overlap/guess.area())
            lowest_child.remove(self.img[0], self.bb)

        return total_reward, end

    def step(self, action):
        self.steps -= 1
        x, y, w, h = float(action[0]), float(action[1]), float(action[2]), float(action[3])
        bb = BoundingBox((x, y, w, h), BoundingBoxType.CENTER)
        reward, terminated = self.calculate_reward((x, y, w, h))
        stoped = self.steps <= 0
        x, y, w, h = bb.get_rect(self.width, self.height)

        obs = self.img
        info = {}

        if (self.render_mode == 'human'):
            """img_copy = self.img.copy()[0]
            cv2.rectangle(img_copy, (x, y), (x+w-1, y+h-1), (127,), 1)
            img_copy = cv2.resize(img_copy, (500, 500), interpolation=cv2.INTER_NEAREST_EXACT)
            print(x, y, w, h, reward)
            cv2.imshow("prediction", img_copy)
            cv2.waitKey(0)"""
            self.render()

        return obs, reward, terminated, stoped, info
    
    def render(self):
        return self.img[0]
        cv2.imshow("square-v6 render", self.img[0])
        cv2.waitKey(0)


if __name__ == "__main__":
    env = gymnasium.make('square-v7', render_mode='human', width=200, height=200)
    env = RescaleAction(env, -1, 1)

    print("check begin")
    check_env(env)
    print("check end")

    """total = 0
    for _ in range(10000):
        env.reset()
        #print(env.env.env.__dict__)
        total += env.env.env.steps

    print(total/10000)"""

    """obs = env.reset()[0]

    for i in range(10):
        rand_action = env.action_space.sample()
        print(rand_action)
        obs, reward, terminated, _, _ = env.step(rand_action)
        print(reward, terminated)
        if (terminated):
            break"""