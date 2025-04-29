import rl.envs.square_v0_env
import rl.envs.square_v1_env
import rl.envs.square_v2_env
import rl.envs.square_v3_env
import rl.envs.square_v4_env
import rl.envs.square_v5_env
import rl.envs.square_v6_env
import rl.envs.square_v7_env
import rl.envs.square_v8_env_discrete
import rl.envs.square_v9_env_discrete
import gymnasium as gym
import cv2
from os import makedirs


ENV = 'square-v9-discrete'
env = gym.make(ENV, width=100, height=100, start_rects=1000)
makedirs(f"thesis_images/envs/{ENV}/", exist_ok=True)

for i in range(10):
    obs, _ = env.reset()
    if obs.shape[0] == 1:
        obs = obs[0]
    cv2.imwrite(f"thesis_images/envs/{ENV}/{i}.jpg", obs)
