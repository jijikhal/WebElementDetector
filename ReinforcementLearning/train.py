import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, PPO, A2C
import os
import square_v2_env_discrete
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


ENV = 'square-v2-discrete'

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
        net_arch=[dict(pi=[256, 128], vf=[256, 128])]     # Actor (pi) and Critic (vf) layers
    )

    model = PPO('CnnPolicy', env, policy_kwargs=policy_kwargs, verbose=True, tensorboard_log=log_dir, device='cuda')
    #model = PPO.load(r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\ReinforcementLearning\logs\v3BiggerNet\best_model\best_model.zip", env=env)
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

    model.learn(total_timesteps=10000000, callback=eval_callback)

if __name__ == '__main__':
    train()