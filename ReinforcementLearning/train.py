import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from stable_baselines3 import SAC, PPO, A2C
import os
import annotator_env
import square_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import datetime
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

ENV = 'square-v0'

def train():
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)

    best_model_path = os.path.join(log_dir, "best_model")

    env = gym.make(ENV)
    #env = SubprocVecEnv([lambda: gym.make('annotator-v0') for i in range(8)])

    model = PPO('CnnPolicy', env, verbose=True, tensorboard_log=log_dir, device='cuda')

    eval_callback = EvalCallback(
        env,
        best_model_save_path=best_model_path,
        log_path=log_dir,
        eval_freq=10000,  # Evaluate every 10000 steps
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=1
    )

    model.learn(total_timesteps=1000000, callback=eval_callback)

# Test using StableBaseline3. Lots of hardcoding for simplicity.
def test_sb3(render=True):

    env = gym.make(ENV, render_mode='human' if render else None)

    # Load model
    model = PPO.load(r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\ReinforcementLearning\logs\03 movingRectDifferentSizesMLP\best_model\best_model.zip", env=env)

    # Run a test
    obs = env.reset()[0]
    terminated = False
    while True:
        action, _ = model.predict(observation=obs, deterministic=False) # Turn on deterministic, so predict always returns the same behavior
        print(action)
        obs, reward, terminated, _, _ = env.step(action)

        print(reward)
        
        obs = env.reset()[0]

        if terminated:
            break

if __name__ == '__main__':

    # Train/test using StableBaseline3
    #train()
    test_sb3()