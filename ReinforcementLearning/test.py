import gymnasium as gym
from gymnasium.wrappers import RescaleAction
from stable_baselines3 import PPO
import square_v3_env_discrete
import square_v7_env

ENV = 'square-v3-discrete'
MODEL = r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\ReinforcementLearning\logs\20250212-000519\best_model\best_model.zip"

def test(render=True):

    env = gym.make(ENV, width=100, height=100, render_mode='human')
    #env = RescaleAction(env, -1, 1) # Normalize Action space

    # Load model
    model = PPO.load(MODEL, env=env)
    total_reward = 0
    total_steps = 0

    for _ in range(1):
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
            print(reward, terminated)

            total_reward += reward

            if terminated or steps >= 100:
                break

    print(total_reward, total_steps)

if __name__ == '__main__':

    # Train/test using StableBaseline3
    test()