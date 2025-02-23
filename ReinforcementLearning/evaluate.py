import gymnasium as gym
from stable_baselines3 import PPO
import square_v7_env_discrete
import cv2
from random import randint
from time import time

ENV = 'square-v7-discrete'
MODEL = r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\ReinforcementLearning\logs\20250223-012229\best_model\best_model.zip"

env = gym.make(ENV, width=100, height=100, render_mode='none')
model = PPO.load(MODEL)

for i in range(10):
    obs = env.reset()[0]
    image = cv2.cvtColor(obs.copy()[0], cv2.COLOR_GRAY2BGR)
    height, width, c = image.shape
    height-=1
    width-=1

    STOP = 8

    predictions = []
    terminated = False
    start = time()

    while not terminated:
        action, _ = model.predict(obs)
        if (action == STOP):
            predictions.append(env.render())
        obs, reward, terminated, _, _ = env.step(action)
        if (terminated):
            break

    print(time()-start)

    for p in predictions:
        cv2.rectangle(image, (int(p[0]*width), int(p[1]*height)), (int(p[2]*width), int(p[3]*height)), (randint(0, 128), randint(0, 128), randint(128, 255)))

    image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("lol",image)
    if cv2.waitKey(0) & 0xFF == 27:
        break
cv2.destroyAllWindows()