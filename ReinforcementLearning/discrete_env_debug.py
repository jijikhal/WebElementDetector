import gymnasium as gym
import square_v8_env_discrete
from square_v8_env_discrete import SHRINK_LEFT, SHRINK_LEFT_SMALL, SHRINK_BOTTOM, SHRINK_BOTTOM_SMALL, SHRINK_RIGHT, SHRINK_RIGHT_SMALL, SHRINK_TOP, SHRINK_TOP_SMALL, STOP
import cv2

ENV = 'square-v8-discrete'

ARROW_LEFT = 2424832
ARROW_UP = 2490368
ARROW_DOWN = 2621440
ARROW_RIGHT = 2555904
KEY_A = 97
KEY_W = 119
KEY_S = 115
KEY_D = 100
SPACE = 32
ESC = 27

ACTION_MAPPING = {
    ARROW_LEFT: SHRINK_LEFT,
    KEY_A: SHRINK_LEFT_SMALL,
    ARROW_UP: SHRINK_TOP,
    KEY_W: SHRINK_TOP_SMALL,
    ARROW_RIGHT: SHRINK_RIGHT,
    KEY_D: SHRINK_RIGHT_SMALL,
    ARROW_DOWN: SHRINK_BOTTOM,
    KEY_S: SHRINK_BOTTOM_SMALL,
    SPACE: STOP
}

def main():
    env = gym.make(ENV, width=84, height=84, render_mode='rgb_array_list', start_rects = 1000)
    while True:  # Episode loop
        print("New episode started")
        obs, info = env.reset()
        terminated = False
        while not terminated:  # Step loop
            whole_render, obs_render = env.render()
            cv2.imshow("Observation Scaled", obs_render)
            cv2.imshow("Observation", obs[0])
            cv2.imshow("Current selection", whole_render)
            key_pressed = cv2.waitKeyEx(0)
            if (key_pressed == ESC):
                return
            if (key_pressed not in ACTION_MAPPING):
                print("Uknown key pressed")
                continue
            action = ACTION_MAPPING[key_pressed]
            obs, reward, terminated, stopped, info = env.step(action)
            print(f"Reward: {reward}")

main()

