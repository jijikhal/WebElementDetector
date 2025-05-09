import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv
from gymnasium.wrappers import TimeLimit
import cv2
from time import time
import wed.rl.envs.square_v9_env_discrete
from wed.rl.envs.square_v9_env_discrete import ObservationType
import numpy as np

ENV = 'square-v9-discrete'
MODEL = r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\wed\rl\logs\20250429-204701\best_model\best_model.zip"

# Recreate the environment
def make_env():
    env = gym.make(ENV, width=84, height=84, render_mode='none', start_rects=100, state_type=ObservationType.STATE_IMAGE_AND_VIEW)  # Use the same ENV as training
    env = TimeLimit(env, max_episode_steps=10000)
    return env

# Create VecEnv
vec_env = DummyVecEnv([make_env])
vec_env = VecMonitor(vec_env)  # Wrap it again

# Load the trained model
model = PPO.load(MODEL, env=vec_env, device='cuda')

for i in range(10):
    obs = vec_env.reset()
    inner_state = None
    image = cv2.cvtColor(vec_env.envs[0].unwrapped.preprocessed.copy(), cv2.COLOR_GRAY2BGR)
    image2 = vec_env.envs[0].unwrapped.base_img.copy()
    #image = cv2.cvtColor(obs[0, 0].copy(), cv2.COLOR_GRAY2BGR)
    height, width, c = image.shape
    height-=1
    width-=1

    STOP = 8

    predictions = []
    terminated = False
    start = time()

    steps = 0
    while not terminated:
        steps += 1
        action, inner_state = model.predict(obs, state=inner_state, deterministic=True)
        if (steps > 300):
            action = np.array([STOP])
        bbox = vec_env.envs[0].unwrapped.view
        obs, reward, terminated, _ = vec_env.step(action)
        if (action[0] == STOP):
            cv2.rectangle(image, (int(bbox[0]*width), int(bbox[1]*height)), (int(bbox[2]*width), int(bbox[3]*height)), (0, int(255*max(0, reward[0]/3)), int(255*(1-max(0, reward[0]/3)))))
            cv2.rectangle(image2, (int(bbox[0]*width), int(bbox[1]*height)), (int(bbox[2]*width), int(bbox[3]*height)), (0, int(255*max(0, reward[0]/3)), int(255*(1-max(0, reward[0]/3)))), 3)
            steps = 0
            print(reward)
            #print(bbox)

    print("Took:", time()-start)

    #image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("lol",image)
    cv2.imshow("lol2",image2)
    if cv2.waitKey(0) & 0xFF == 27:
        break
cv2.destroyAllWindows()