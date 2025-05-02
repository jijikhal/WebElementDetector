import torch
import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# This was created with assistance from ChatGPT
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=512)  # We'll update features_dim later
        
        # This part should be identical to SB3's CnnPolicy
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        with torch.no_grad():
            sample_image = torch.zeros(1, *observation_space["image"].shape)
            cnn_output_dim = self.cnn(sample_image).shape[1]
        
        # Fully connected layer after CNN
        self.linear = nn.Sequential(
            nn.Linear(cnn_output_dim+observation_space["view"].shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )

        # Final feature dim is cnn + bbox
        self._features_dim = 512

    def forward(self, observations):
        cnn_out = self.cnn(observations["image"].float() / 255.0)  # Normalize image
        bbox_out = observations["view"]  # Already float, no need for change
        return self.linear(torch.cat([cnn_out, bbox_out], dim=1))
