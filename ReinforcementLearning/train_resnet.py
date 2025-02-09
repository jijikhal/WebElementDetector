from pydoc import cli
import gymnasium as gym
from gymnasium.wrappers import RescaleAction
from stable_baselines3 import SAC, PPO, A2C
from stable_baselines3.common.policies import ActorCriticPolicy
import os
import square_v2_env_discrete
from stable_baselines3.common.vec_env import SubprocVecEnv
import datetime
from stable_baselines3.common.callbacks import EvalCallback
import torch
from torchvision import models
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

class GrayscaleToRGB(nn.Module):
    def __init__(self):
        super(GrayscaleToRGB, self).__init__()
        self.linear = nn.Conv2d(1, 3, kernel_size=1)  # Learnable 1x1 convolution

    def forward(self, x):
        return self.linear(x)

class PretrainedResNetFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, output_dim=512):
        super(PretrainedResNetFeatureExtractor, self).__init__(observation_space, features_dim=output_dim)
        
        self.grayscale_to_rgb = nn.Sequential(GrayscaleToRGB())

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
    
class CustomPPOPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPPOPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=PretrainedResNetFeatureExtractor,
            features_extractor_kwargs=dict(output_dim=512),
        )

ENV = 'square-v2-discrete'

def train():
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)

    best_model_path = os.path.join(log_dir, "best_model")

    env = gym.make(ENV, width=100, height=100)
    #env = RescaleAction(env, min_action=-1, max_action=1) # Normalize Action space

    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 128], vf=[256, 128])]     # Actor (pi) and Critic (vf) layers
    )

    model = PPO(CustomPPOPolicy, env, policy_kwargs=policy_kwargs, verbose=True, tensorboard_log=log_dir, device='cuda', gamma=0.95, clip_range=0.15)
    #model = PPO.load(r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\ReinforcementLearning\logs\v5ResNetFineTune\best_model\best_model.zip", env=env)
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