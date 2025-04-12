import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, PPO, A2C
import os
import square_v2_env_discrete
import square_v5_env_discrete
import square_v7_env_discrete
import square_v8_env_discrete
import datetime
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack, SubprocVecEnv, VecMonitor
from sb3_contrib import RecurrentPPO
from torch import nn
from gymnasium.wrappers import TimeLimit
import math

ENV = 'square-v8-discrete'

def exp_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return math.exp(-2*(1-progress_remaining))*initial_value*0.6

    return func

## Taken from rl-zoo
def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func

def make_env(**kwargs):
    def _init():
        env = gym.make(ENV, width=84, height=84, **kwargs)
        env = TimeLimit(env, max_episode_steps=1000)
        return env
    return _init

def train():
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #os.makedirs(log_dir, exist_ok=True)

    best_model_path = os.path.join(log_dir, "best_model")

    n_envs = 6
    vec_env = SubprocVecEnv([make_env(name=f"Train env {i}", start_rects=3) for i in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    policy_kwargs = dict(
        net_arch=[dict(pi=[512], vf=[512])],     # Actor (pi) and Critic (vf) layers
        activation_fn=nn.ReLU,
        normalize_images=False
    )

    model = PPO('CnnPolicy', vec_env, policy_kwargs=policy_kwargs, verbose=True, tensorboard_log=log_dir, device='cuda',
                batch_size=512,
                n_steps=256,
                gamma=0.999,
                learning_rate = linear_schedule(2.5e-4),
                ent_coef=0.01,
                clip_range=linear_schedule(0.1),
                n_epochs=4,
                gae_lambda=0.95,
                vf_coef=0.5,
                )
    print(sum(p.numel() for p in model.policy.parameters()))

    eval_env = DummyVecEnv([make_env(name="Eval env", start_rects=1000)])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)
    eval_env.obs_rms = vec_env.obs_rms

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_path,
        log_path=log_dir,
        eval_freq=10000,  # Evaluate every 10000 steps
        deterministic=True,
        render=False,
        n_eval_episodes=20,
        verbose=1,
    )

    model.learn(total_timesteps=20_000_000, callback=eval_callback, log_interval=10)
    vec_env.save(os.path.join(log_dir, "vec_normalize.pkl"))

if __name__ == '__main__':
    train()