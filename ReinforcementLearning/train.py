import gymnasium as gym
from gymnasium.wrappers.rescale_action import RescaleAction
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, PPO, A2C
import os
import annotator_env
import square_v2_env
import square_v3_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import datetime
from stable_baselines3.common.callbacks import EvalCallback

ENV = 'square-v3'

def train():
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)

    best_model_path = os.path.join(log_dir, "best_model")

    env = gym.make(ENV)
    env = RescaleAction(env, -1, 1) # Normalize Action space
    # Observation space normalization is done by SB3 for CNN
    #env = SubprocVecEnv([lambda: gym.make(ENV) for i in range(8)])

    #model = PPO('CnnPolicy', env, verbose=True, tensorboard_log=log_dir, device='cuda')
    model = PPO.load(r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\ReinforcementLearning\logs\v2CNNnew\best_model\best_model.zip", env=env)

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

if __name__ == '__main__':
    train()