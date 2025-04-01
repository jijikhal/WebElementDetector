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
from torch import nn
from gymnasium.wrappers import TimeLimit

ENV = 'square-v8-discrete'

## Taken from rl-zoo
def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func

def train():
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #os.makedirs(log_dir, exist_ok=True)

    best_model_path = os.path.join(log_dir, "best_model")

    n_envs = 4
    env = gym.make(ENV, width=84, height=84)
    vec_env = SubprocVecEnv([lambda: TimeLimit(env, max_episode_steps=1000) for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    vec_env = VecFrameStack(vec_env, 4, "first")

    policy_kwargs = dict(
        net_arch=[dict(pi=[512], vf=[512])],     # Actor (pi) and Critic (vf) layers
        activation_fn=nn.ReLU,
        normalize_images=False
    )

    model = PPO('CnnPolicy', vec_env, policy_kwargs=policy_kwargs, verbose=True, tensorboard_log=log_dir, device='cuda',
                batch_size=512,
                n_steps=128,
                gamma=0.999,
                learning_rate = linear_schedule(2.5e-4),
                ent_coef=0.01,
                clip_range=linear_schedule(0.1),
                n_epochs=4,
                gae_lambda=0.95,
                vf_coef=0.5,
                )
    print(sum(p.numel() for p in model.policy.parameters()))

    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path=best_model_path,
        log_path=log_dir,
        eval_freq=10000,  # Evaluate every 10000 steps
        deterministic=True,
        render=False,
        n_eval_episodes=20,
        verbose=1,
    )

    model.learn(total_timesteps=10_000_000, callback=eval_callback, log_interval=10)
    vec_env.save(os.path.join(log_dir, "vec_normalize.pkl"))

if __name__ == '__main__':
    train()