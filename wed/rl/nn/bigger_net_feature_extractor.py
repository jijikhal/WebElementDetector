import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import gymnasium as gym

# Custom CNN feature extractor
class BiggerNetExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        # Initialize the base class
        super(BiggerNetExtractor, self).__init__(observation_space, features_dim)
        
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