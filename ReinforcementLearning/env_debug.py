import gymnasium
from bounding_box import BoundingBox, BoundingBoxType
import square_v7_env
import cv2
import numpy as np

last_x = -1
last_y = -1
env = gymnasium.make('square-v7', render_mode='human', height=200, width=200)
# Change None to some number to make it determinisitc
env.reset(seed=None)

def draw_circle(event,x,y,flags,param):
    global last_x, last_y, env
    if event == cv2.EVENT_LBUTTONDOWN:
        if (last_x == -1):
            last_x = x
            last_y = y
        else:
            bb = BoundingBox((min(last_x, x)/200, min(last_y, y)/200, abs(last_x-x)/200, abs(last_y-y)/200), BoundingBoxType.TOP_LEFT)
            action = np.array(bb.get_bb_middle())
            print(last_x, last_y, x, y)
            print(action)
            obs, reward, terminated, _, _ = env.step(action)
            print("reward: ", reward, " terminated: ", terminated)
            last_x = -1
            last_y = -1
            print(env.env.env.bb)
    
if __name__ == "__main__":
    cv2.namedWindow('square-v7 render')
    cv2.setMouseCallback('square-v7 render', draw_circle)

    while(1):
        frame = env.render()
        cv2.imshow('square-v7 render', frame)

        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break

    env.close()
    cv2.destroyAllWindows()
