import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from stable_baselines3 import SAC, A2C
import os
import annotator_env

def train():
    model_dir = "models"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make('annotator-v0')

    model = A2C('CnnPolicy', env, verbose=True, tensorboard_log=log_dir, device='cuda')

    TIMESTEPS = 1000
    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False) # train
        model.save(f"{model_dir}/a2c") # Save a trained model every TIMESTEPS

# Test using StableBaseline3. Lots of hardcoding for simplicity.
def test_sb3(render=True):

    env = gym.make('annotator-v0', render_mode='human' if render else None)

    # Load model
    model = A2C.load('models/a2c', env=env)

    # Run a test
    obs = env.reset()[0]
    terminated = False
    while True:
        action, _ = model.predict(observation=obs, deterministic=False) # Turn on deterministic, so predict always returns the same behavior
        print(action)
        obs, reward, terminated, _, _ = env.step(action)

        print(reward)

        if terminated:
            break

if __name__ == '__main__':

    # Train/test using StableBaseline3
    #train()
    test_sb3()