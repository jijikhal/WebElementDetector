import gymnasium as gym
from gymnasium.wrappers.rescale_action import RescaleAction
from stable_baselines3 import SAC, PPO, A2C
import annotator_env
import square_v3_env

ENV = 'square-v3'
MODEL = r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\ReinforcementLearning\logs\20241030-223057\best_model\best_model.zip"

def test(render=True):

    env = gym.make(ENV, render_mode='human' if render else None)
    env = RescaleAction(env, -1, 1) # Normalize Action space

    # Load model
    model = PPO.load(MODEL, env=env)

    while True:
        # Run a test
        obs = env.reset()[0]
        terminated = False
        steps = 0
        while True:
            steps += 1
            action, _ = model.predict(observation=obs, deterministic=True) # Turn on deterministic, so predict always returns the same behavior
            print(action)
            obs, reward, terminated, _, _ = env.step(action)

            print(reward)

            if terminated or steps >= 10:
                break

if __name__ == '__main__':

    # Train/test using StableBaseline3
    test()