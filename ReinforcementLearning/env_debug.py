import gymnasium
from bounding_box import BoundingBox, BoundingBoxType
import square_v6_env
import cv2
import numpy as np

last_x = -1
last_y = -1
env = gymnasium.make('square-v6', render_mode='human')
env.reset(seed=0)

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
    
if __name__ == "__main__":
    cv2.namedWindow('square-v6 render')
    cv2.setMouseCallback('square-v6 render', draw_circle)

    while(1):
        frame = env.render()
        cv2.imshow('square-v6 render', frame)

        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break

    env.close()
    cv2.destroyAllWindows()
