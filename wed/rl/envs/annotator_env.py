import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
from annotator import Annotator
import numpy as np
from wed.bounding_box import BoundingBox, RectF, RectI, BoundingBoxType
import cv2
# https://www.youtube.com/watch?v=AoGRjPt-vms

register(
    id='annotator-v0',
    entry_point='annotator_env:AnnotatorEnv'
)

class AnnotatorEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps':1} 
    def __init__(self, height: int = 100, width: int = 100, folder: str = "images", render_mode=None) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.render_mode = render_mode
        self.steps=0

        self.annotator = Annotator(height=height, width=width, folder=folder)

        self.action_space = spaces.Box(low=0, high=np.array([1.0, 1.0, 1.0, 1.0]), shape=(4,), dtype=np.float32)

        self.observation_space = spaces.Box(low=0, high=255, shape=(1, height, width), dtype=np.uint8)

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        self.annotator.reset(seed=seed)
        obs = self.annotator.img
        self.steps = 0
        info = {}

        if (self.render_mode == 'human'):
            self.render()

        return obs, info
    
    def calculate_reward(self, rect: RectF) -> tuple[float, bool]:
        img = self.annotator.img[0]

        total_reward = 0

        ih, iw = self.height, self.width

        bb = BoundingBox(rect, BoundingBoxType.CENTER)
        x, y, w, h = bb.get_rect(self.width, self.height)
        
        if (w < 5 or h < 5):
            return -100, False

        cutout = img[y:y+h, x:x+w]
        terminated = w*h/(iw*ih) > 0.9

        # count edge intersections
        edge = cutout.copy()
        edge[1:h-1, 1:w-1] = 0
        total_edge_pixels = w*2+h*2-4
        white_edge_pixels = np.count_nonzero(edge > 0)
        percentage_cross = white_edge_pixels/total_edge_pixels
        if (white_edge_pixels > 0):
            total_reward -= 100*percentage_cross

        # Count enclosed components
        _, binary = cv2.threshold(cutout, 254, 255, cv2.THRESH_BINARY)
        retval, _ = cv2.connectedComponents(binary)

        if (5 > retval-1 > 0):
            total_reward += 1
        else:
            total_reward -= 10


        # Count distance from edges
        white_pixel_coords = np.column_stack(np.where(cutout == 255))
        if white_pixel_coords.size == 0:
            return total_reward, terminated
        
        # Get min and max for x and y
        min_y, min_x = white_pixel_coords.min(axis=0)
        max_y, max_x = white_pixel_coords.max(axis=0)

        top_dist = (min_y)/ih
        bottom_dist = (cutout.shape[0] - 1 - max_y)/ih
        left_dist = (min_x)/iw
        right_dist = (cutout.shape[1] - 1 - max_x)/iw

        for d in [top_dist, bottom_dist, left_dist, right_dist]:
            if (d < 0.01):
                total_reward += 1-d*100

        return total_reward, terminated

    def step(self, action):
        self.steps += 1
        x, y, w, h = float(action[0]), float(action[1]), float(action[2]), float(action[3])
        reward, terminated = self.calculate_reward((x, y, w, h))
        self.annotator.perform_action((x, y, w, h))
        stoped = self.steps >= 100 

        obs = self.annotator.img
        info = {}

        if (self.render_mode == 'human'):
            self.render()

        return obs, reward, terminated, stoped, info
    
    def render(self):
        self.annotator.render()


if __name__ == "__main__":
    env = gym.make('annotator-v0', render_mode='human')

    #print("check begin")
    #check_env(env.unwrapped)
    #print("check end")

    obs = env.reset()[0]

    for i in range(10):
        rand_action = env.action_space.sample()
        print(rand_action)
        obs, reward, terminated, _, _ = env.step(rand_action)
        print(reward, terminated)