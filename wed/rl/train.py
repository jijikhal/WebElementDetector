import gymnasium as gym
from stable_baselines3 import SAC, PPO, A2C
import os
import wed.rl.envs.square_v9_env_discrete
from wed.rl.envs.square_v9_env_discrete import ObservationType
import datetime
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from torch import nn
from gymnasium.wrappers import TimeLimit
from wed.rl.nn.bigger_net_feature_extractor import BiggerNetExtractor
from wed.rl.nn.resnet_feature_extractor import ResNetExtractor
from wed.rl.nn.combined_feature_extractor import CustomCombinedExtractor
from wed.rl.nn.resnet_combined_feature_extractor import ResNetCombinedExtractor
from wed.rl.nn.squeezenet_combined_feature_extractor import SqueezeNetCombinedExtractor
from wed.rl.nn.schedules import linear_schedule

ENV = 'square-v9-discrete'

def make_env(**kwargs):
    def _init():
        env = gym.make(ENV, width=84, height=84, **kwargs)
        env = TimeLimit(env, max_episode_steps=1000)
        return env
    return _init

def train():
    log_dir = os.path.join(r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\wed\rl\logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    #os.makedirs(log_dir, exist_ok=True)

    best_model_path = os.path.join(log_dir, "best_model")

    n_envs = 6
    vec_env = SubprocVecEnv([make_env(name=f"Train env {i}", start_rects=3, state_type=ObservationType.STATE_IMAGE_AND_VIEW, dataset_folder=r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\wed\yolo\dataset\images\train") for i in range(n_envs)])
    vec_env = VecMonitor(vec_env)

    policy_kwargs_default = dict(
        net_arch=[dict(pi=[1024, 1024], vf=[1024, 1024])],
        activation_fn=nn.ReLU,
        normalize_images=False,
        features_extractor_class=SqueezeNetCombinedExtractor,
    )

    policy_kwargs_bigger_net = dict(
        features_extractor_class=BiggerNetExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[dict(pi=[256, 128], vf=[256, 128])],
        ortho_init=False,
        activation_fn=nn.Tanh
    )

    policy_kwargs_resnet = dict(
        features_extractor_class=ResNetExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[dict(pi=[256, 128], vf=[256, 128])],
    )

    policy_kwargs_resnet_combined = dict(
        features_extractor_class=ResNetCombinedExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[dict(pi=[256, 128], vf=[256, 128])],
    )

    model = PPO('MultiInputPolicy', vec_env, policy_kwargs=policy_kwargs_default, verbose=True, tensorboard_log=log_dir, device='cuda',
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
    print(model.policy)

    eval_env = DummyVecEnv([make_env(name="Eval env", start_rects=1000, state_type=ObservationType.STATE_IMAGE_AND_VIEW, dataset_folder=r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\wed\yolo\dataset\images\val")])
    eval_env = VecMonitor(eval_env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_path,
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=20,
        verbose=1,
    )

    model.learn(total_timesteps=20_000_000, callback=eval_callback, log_interval=10)

if __name__ == '__main__':
    train()