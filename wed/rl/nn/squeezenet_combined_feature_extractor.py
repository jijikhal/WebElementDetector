import torch
from torchvision import models
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import gymnasium as gym

# This was created with assistance from ChatGPT


class SqueezeNetCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim=1):
        super(SqueezeNetCombinedExtractor, self).__init__(observation_space, features_dim=features_dim)
        # Load pretrained ResNet
        sqeezenet = models.squeezenet1_1(pretrained=True)
        self.sqeezenet = nn.Sequential(nn.Conv2d(1, 3, kernel_size=1), sqeezenet.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

        with torch.no_grad():
            dummy_input = torch.zeros(1, *observation_space["image"].shape)
            resnet_output_dim = self.sqeezenet(dummy_input).shape[1]

        # Going from 512 to 4096 probably does not make much senese, but I discovered it only
        # at the end when creating the diagrams :)
        self.linear = nn.Linear(resnet_output_dim, 4096)

        view_dim = observation_space["view"].shape[0]
        self._features_dim = 4096+view_dim

    def forward(self, observations):
        # Assuming input is a 3D tensor (C x H x W)
        cnn_out = self.linear(self.sqeezenet(observations["image"].float() / 255.0))
        bbox_out = observations["view"]
        return torch.cat([cnn_out, bbox_out], dim=1)
