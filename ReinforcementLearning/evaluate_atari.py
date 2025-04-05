import gymnasium as gym
import torch
import os
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecFrameStack, VecNormalize, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.wrappers import TimeLimit
import cv2
from time import time
import square_v8_env_discrete
import numpy as np

ENV = 'square-v8-discrete'
MODEL = r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\ReinforcementLearning\logs\20250405-010412\best_model\best_model.zip"
NORM = r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\ReinforcementLearning\logs\20250405-010412\vec_normalize.pkl"

# Recreate the environment
def make_env():
    env = gym.make(ENV, width=84, height=84, render_mode='none')  # Use the same ENV as training
    env = TimeLimit(env, max_episode_steps=1000)
    return env

# Create VecEnv
vec_env = DummyVecEnv([make_env])
vec_env = VecMonitor(vec_env)  # Wrap it again
vec_env = VecNormalize.load(NORM, vec_env)  # Load normalization

# Set evaluation mode to avoid updating normalization stats
vec_env.training = False
vec_env.norm_reward = False

# Load the trained model
model = RecurrentPPO.load(MODEL, env=vec_env, device='cuda')

for i in range(10):
    obs = vec_env.reset()
    inner_state = None
    image = cv2.cvtColor(obs[0, 0].copy(), cv2.COLOR_GRAY2BGR)
    height, width, c = image.shape
    height-=1
    width-=1

    STOP = 8

    predictions = []
    terminated = False
    start = time()

    steps = 0
    while not terminated:
        steps += 1
        action, inner_state = model.predict(obs, state=inner_state, deterministic=True)
        if (steps > 30):
            action = np.array([STOP])
        bbox = vec_env.envs[0].unwrapped.view
        obs, reward, terminated, _ = vec_env.step(action)
        if (action[0] == STOP):
            cv2.rectangle(image, (int(bbox[0]*width), int(bbox[1]*height)), (int(bbox[2]*width), int(bbox[3]*height)), (0, int(255*max(0, reward/3)), int(255*(1-max(0, reward/3)))))
            steps = 0
            print(reward)
            print(bbox)

    print("Took:", time()-start)

    image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("lol",image)
    if cv2.waitKey(0) & 0xFF == 27:
        break
cv2.destroyAllWindows()