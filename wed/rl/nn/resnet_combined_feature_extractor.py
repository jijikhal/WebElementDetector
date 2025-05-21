# Feature extractor mentioned in Subsection 4.17.1
import torch
from torchvision import models
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import gymnasium as gym

# This was created with assistance from ChatGPT


class ResNetCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim=1):
        super(ResNetCombinedExtractor, self).__init__(observation_space, features_dim=features_dim)
        # Load pretrained ResNet
        resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(nn.Conv2d(1, 3, kernel_size=1), *list(resnet.children())[:-1], nn.Flatten())

        with torch.no_grad():
            dummy_input = torch.zeros(1, *observation_space["image"].shape)
            resnet_output_dim = self.resnet(dummy_input).shape[1]

        view_dim = observation_space["view"].shape[0]
        self._features_dim = resnet_output_dim+view_dim

    def forward(self, observations):
        # Assuming input is a 3D tensor (C x H x W)
        cnn_out = self.resnet(observations["image"].float() / 255.0)
        bbox_out = observations["view"]
        return torch.cat([cnn_out, bbox_out], dim=1)
