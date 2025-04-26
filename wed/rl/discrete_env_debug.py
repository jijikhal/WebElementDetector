import gymnasium as gym
import envs.square_v9_env_discrete
from envs.square_v9_env_discrete import Action, ObservationType
import cv2

ENV = 'square-v9-discrete'

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
    ARROW_LEFT: Action.SHRINK_LEFT,
    KEY_A: Action.SHRINK_LEFT_SMALL,
    ARROW_UP: Action.SHRINK_TOP,
    KEY_W: Action.SHRINK_TOP_SMALL,
    ARROW_RIGHT: Action.SHRINK_RIGHT,
    KEY_D: Action.SHRINK_RIGHT_SMALL,
    ARROW_DOWN: Action.SHRINK_BOTTOM,
    KEY_S: Action.SHRINK_BOTTOM_SMALL,
    SPACE: Action.STOP
}

def main():
    env = gym.make(ENV, width=150, height=150, render_mode='rgb_array_list', start_rects = 1000, state_type=ObservationType.STATE_IMAGE_AND_VIEW, padding=0.00)
    while True:  # Episode loop
        print("New episode started")
        obs, info = env.reset()
        terminated = False
        while not terminated:  # Step loop
            whole_render, obs_render = env.render()
            cv2.imshow("Observation Scaled", obs_render)
            if (isinstance(obs, dict)):
                cv2.imshow("Observation", obs["image"][0])
            else:
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

