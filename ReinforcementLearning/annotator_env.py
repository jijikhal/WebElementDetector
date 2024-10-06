import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
from annotator import Annotator
import numpy as np
# https://www.youtube.com/watch?v=AoGRjPt-vms

register(
    id='annotator-v0',
    entry_point='annotator_env:AnnotatorEnv'
)

class AnnotatorEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps':1} 
    def __init__(self, height: int = 270, width: int = 480, folder: str = "images", render_mode=None) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.render_mode = render_mode
        self.steps=0

        self.annotator = Annotator(height=height, width=width, folder=folder)

        self.action_space = spaces.Box(low=0, high=np.array([1.0, 1.0, 1.0, 1.0]), shape=(4,), dtype=np.float32)

        self.observation_space = spaces.Box(low=0, high=255, shape=(height, width), dtype=np.uint8)

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        self.annotator.reset(seed=seed)
        obs = self.annotator.img
        info = {}

        if (self.render_mode == 'human'):
            self.render()

        return obs, info
    
    def step(self, action):
        self.steps += 1
        x1, y1, x2, y2 = float(action[0]), float(action[1]), float(action[2]), float(action[3])
        target_reached = self.annotator.perform_action(x1, y1, x2, y2)
        reward = abs(x1-x2)*abs(y1-y2)
        terminated = False
        stoped = self.steps >= 10

        obs = self.annotator.img
        info = {}

        if (self.render_mode == 'human'):
            self.render()

        return obs, reward, terminated, stoped, info
    
    def render(self):
        self.annotator.render()


if __name__ == "__main__":
    env = gym.make('annotator-v0', render_mode='human')

    print("check begin")
    #check_env(env.unwrapped)
    print("check end")

    obs = env.reset()[0]

    for i in range(10):
        rand_action = env.action_space.sample()
        print(rand_action)
        obs, reward, terminated, _, _ = env.step(rand_action)