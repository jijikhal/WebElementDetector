import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, PPO, A2C
import os
import square_v2_env_discrete
import square_v5_env_discrete
import square_v7_env_discrete
from stable_baselines3.common.vec_env import SubprocVecEnv
import datetime
from stable_baselines3.common.callbacks import EvalCallback
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device
from torch import nn

# Custom CNN feature extractor
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        # Initialize the base class
        super(CustomCNN, self).__init__(observation_space, features_dim)
        
        # Assumes input is 1 channel, 100x100 grayscale image
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),  # Output: 32 x 23 x 23
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),  # Output: 64 x 10 x 10
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0), # Output: 128 x 8 x 8
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the output dimension dynamically
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
        
        # Fully connected layer after CNN
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


ENV = 'square-v7-discrete'

## Taken from rl-zoo
def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func

def custom_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        total_timesteps = 10_000_000  # Example: Total training steps
        current_step = int((1 - progress_remaining) * total_timesteps)
        if current_step < 200_000:
            return initial_value * (1 - current_step / 200_000)
        else:
            return initial_value * 0.1  # Hold at 10% of initial value after 200k steps

    return func

def train():
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #os.makedirs(log_dir, exist_ok=True)

    best_model_path = os.path.join(log_dir, "best_model")

    env = gym.make(ENV)
    # Observation space normalization is done by SB3 for CNN
    #env = SubprocVecEnv([lambda: gym.make(ENV) for i in range(8)])

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),  # Set the output feature dimension of the CNN
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],     # Actor (pi) and Critic (vf) layers
        ortho_init=False,
        activation_fn=nn.ReLU
    )

    model = PPO('CnnPolicy', env, policy_kwargs=policy_kwargs, verbose=True, tensorboard_log=log_dir, device='cuda',
                batch_size=32,
                n_steps=2048,
                gamma=0.99,
                learning_rate=custom_schedule(2.6226217364486832e-06),
                ent_coef=2.3405243352330302e-05,
                clip_range=0.3,
                n_epochs=20,
                gae_lambda=0.9,
                max_grad_norm=0.3,
                vf_coef=0.8968584303991769,
                )
    old_model = PPO.load(r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\ReinforcementLearning\logs\20250223-175015\best_model\best_model.zip", env=env)
    model.policy.load_state_dict(old_model.policy.state_dict())
    #model.learning_rate = 4.6226217364486832e-06
    print(model.policy)
    print(sum(p.numel() for p in model.policy.parameters()))

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

    model.learn(total_timesteps=10_000_000, callback=eval_callback)

if __name__ == '__main__':
    train()