import torch
from torchvision import models
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

class ResNetExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, output_dim=512):
        super(ResNetExtractor, self).__init__(observation_space, features_dim=output_dim)
        
        self.grayscale_to_rgb = nn.Conv2d(1, 3, kernel_size=1)

        # Load pretrained ResNet
        resnet = models.resnet18(pretrained=True)
        
        # Remove the final fully connected layer (classifier)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Retain all layers except the last one
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, *observation_space.shape)  # Assuming shape is (C, H, W)
            dummy_output = self.feature_extractor(self.grayscale_to_rgb(dummy_input))
            self._features_dim = int(torch.prod(torch.tensor(dummy_output.shape[1:])))
            print("size", int(torch.prod(torch.tensor(dummy_output.shape[1:]))), observation_space.shape)

    def forward(self, observations):
        # Assuming input is a 3D tensor (C x H x W)
        x = self.feature_extractor(self.grayscale_to_rgb(observations))
        return torch.flatten(x, start_dim=1)