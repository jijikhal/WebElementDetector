import gymnasium as gym
from stable_baselines3 import PPO
import square_v8_env_discrete
import cv2
from random import randint
from time import time

ENV = 'square-v8-discrete'
MODEL = r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\ReinforcementLearning\logs\20250327-115651\best_model\best_model.zip"

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

    steps = 0
    while not terminated:
        steps += 1
        action, _ = model.predict(obs, deterministic=True)
        if (steps > 30):
            action = STOP
        bbox = env.render()
        obs, reward, terminated, _, _ = env.step(action)
        if (action == STOP):
            cv2.rectangle(image, (int(bbox[0]*width), int(bbox[1]*height)), (int(bbox[2]*width), int(bbox[3]*height)), (0, int(255*max(0, reward/3)), int(255*(1-max(0, reward/3)))))
            steps = 0
            print(reward)
            print(bbox)

    print("Took:", time()-start)

    image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("lol",image)
    if cv2.waitKey(0) & 0xFF == 27:
        break
cv2.destroyAllWindows()