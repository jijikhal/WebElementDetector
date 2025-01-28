import gymnasium as gym
from gymnasium.wrappers import RescaleAction
from stable_baselines3 import SAC, PPO, A2C
import annotator_env
import square_v6_env

ENV = 'square-v6'
MODEL = r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\ReinforcementLearning\logs\v6200x200_25_iou\best_model\best_model.zip"

def test(render=True):

    env = gym.make(ENV, width=200, height=200, reward_func=square_v6_env.PUNISH_IOU, render_mode='none')
    env = RescaleAction(env, -1, 1) # Normalize Action space

    # Load model
    model = PPO.load(MODEL, env=env)
    total_reward = 0
    total_steps = 0

    for _ in range(5000):
        # Run a test
        obs = env.reset()[0]
        terminated = False
        steps = 0
        while True:
            steps += 1
            total_steps += 1
            action, _ = model.predict(observation=obs, deterministic=True) # Turn on deterministic, so predict always returns the same behavior
            #print(action)
            obs, reward, terminated, _, _ = env.step(action)

            total_reward += reward

            if terminated or steps >= 10:
                break

    print(total_reward, total_steps)

if __name__ == '__main__':

    # Train/test using StableBaseline3
    test()