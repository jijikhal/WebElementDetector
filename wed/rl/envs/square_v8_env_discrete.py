import random
from tracemalloc import start
import gymnasium
from gymnasium import spaces
from gymnasium.envs.registration import register
from wed.bounding_box import BoundingBox, RectF, RectI, BoundingBoxType
import numpy as np
import cv2
from cv2.typing import MatLike
from stable_baselines3.common.env_checker import check_env
from os import listdir
from os.path import isfile, join
from time import time
import datetime
from enum import IntEnum
from wed.rl.envs.common import Action, ObservationType

register(
    id='square-v8-discrete',
    entry_point='wed.rl.envs.square_v8_env_discrete:SquareEnv'
)

def draw_rect(bb: RectI, image: MatLike) -> None:
    x, y, w, h = bb
    cv2.rectangle(image, (x, y), (x+w-1, y+h-1), (255,), 1)

def find_bounding_boxes(img: MatLike) -> list[BoundingBox]:
    copy_for_show = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.bilateralFilter(gray, 21, 50, 0)
    binary = cv2.Canny(smoothed, 22, 22, L2gradient=True)
    big_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    big_count, _, big_stats, _ = cv2.connectedComponentsWithStats(big_close, connectivity=8)
    img_h, img_w, _ = img.shape

    result: list[BoundingBox] = [BoundingBox((0, 0, 1, 1), BoundingBoxType.TWO_CORNERS)]

    for i in range(1, big_count):
        x, y, w, h, area = big_stats[i]
        if (min(w,h) < 10 or max(w, h) < 20):
            cv2.rectangle(copy_for_show, (x, y), (x + w + 1, y + h + 1), (0, 0, 255), 1)
            continue
        result.append(BoundingBox((x/img_w, y/img_h, (w+1)/img_w, (h+1)/img_h), BoundingBoxType.TOP_LEFT))
        cv2.rectangle(copy_for_show, (x, y), (x + w + 1, y + h + 1), (0, 255, 0), 2)

    merged: list[BoundingBox] = []
    used = set()
    result.sort(key=lambda x: x.area())
    for b in result:
        if (b in used):
            continue
        used.add(b)
        to_merge_with = [x for x in result if b.is_intersecting(x) and x not in used]
        m = b
        for t in to_merge_with:
            used.add(t)
            m = b.merge(t)
        merged.append(m)

    for b in merged:
        x1, y1, x2, y2 = b.get_bb_corners()
        cv2.rectangle(copy_for_show, (round(x1*img_w), round(y1*img_h)), (round(x2*img_w), round(y2*img_h)), (255, 0, 0), 2)

    merged.sort(key=lambda x: x.area(), reverse=True)

    #cv2.imshow("img", copy_for_show)
    #cv2.waitKey(0)
    
    return merged

class SquareEnv(gymnasium.Env):
    metadata = {'render_modes': ['human','none', 'rgb_array_list', 'rgb_array']} 
    def __init__(self, height: int = 100, width: int = 100, render_mode=None, dataset_folder: str = r"rl\dataset_big", start_rects: int = 3, name: str = "env", state_type = ObservationType.STATE_IMAGE_ONLY, padding: float = 0.00) -> None:
        super().__init__()
        self.height: int = height
        self.width: int = width
        self.render_mode = render_mode
        self.image_paths = [join(dataset_folder, f) for f in listdir(dataset_folder) if isfile(join(dataset_folder, f))]
        self.base_img_path = random.choice(self.image_paths)
        self.base_img = cv2.imread(self.base_img_path)
        self.max_bbs = start_rects
        self.ground_truth_labels: list[BoundingBox] = find_bounding_boxes(self.base_img)[:self.max_bbs]
        self.view: list[float] = [0.0, 0.0, 1.0, 1.0]
        self.last_reward = 0
        self.padding = padding
        self._padding_pixels: tuple[int, int] = (0,0)
        self.preprocessed: MatLike = self._preprocess_img()
        self.reward_archive: list[float] = []
        self.current_best_bb = None
        self.name = name
        self._state_type = state_type

        self.action_space = spaces.Discrete(9)
        if (self._state_type == ObservationType.STATE_IMAGE_ONLY):
            self.observation_space = spaces.Box(low=0, high=255, shape=(1, height, width), dtype=np.uint8)
        elif (self._state_type == ObservationType.STATE_IMAGE_AND_VIEW):
            self.observation_space = spaces.Dict({
                "image": spaces.Box(low=0, high=255, shape=(1, height, width), dtype=np.uint8),
                "view": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
            })
        else:
            raise Exception("Unknown State type")

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        random.seed(seed)
        np.random.seed(seed)

        self.base_img_path = random.choice(self.image_paths)
        self.base_img = cv2.imread(self.base_img_path)
        self.view = [0, 0, 1, 1]

        # Curriculum learning
        if (len(self.reward_archive) >= 100):
            avg = sum(self.reward_archive)/len(self.reward_archive)
            if (avg > 0.9):
                self.max_bbs += 1
                print(f"Increased difficulty of env {self.name} to {self.max_bbs}")
                try:
                    with open("difficulty_log.txt", "a") as f:
                        f.write(f"Increased difficulty of env {self.name} to {self.max_bbs} at {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}\n")
                except:
                    pass

            self.reward_archive = []


        self.ground_truth_labels = find_bounding_boxes(self.base_img)[:self.max_bbs]
        self.preprocessed = self._preprocess_img()

        self.last_reward = 0
        self.last_reward = self.calculate_reward_dense(BoundingBox(self.view, BoundingBoxType.TWO_CORNERS), False)[0]

        obs = self.get_observation()
        info = {}

        if (self.render_mode == 'human'):
            self.render()

        return obs, info
    
    def _preprocess_img(self) -> MatLike:
        img_h, img_w, _ = self.base_img.shape
        img = np.zeros((img_h, img_w), dtype=np.uint8)
        for l in self.ground_truth_labels:
            x1, y1, x2, y2 = l.get_bb_corners()
            cv2.rectangle(img, (round(x1*img_w), round(y1*img_h)), (round(x2*img_w)-1, round(y2*img_h)-1), (255,), 1)

        padding_x = round(img_w*self.padding)
        padding_y = round(img_h*self.padding)
        self._padding_pixels = (padding_x, padding_y)

        padded = np.pad(img, ((padding_y, padding_y),(padding_x, padding_x)), 'constant', constant_values=0)
        return padded
    
    def get_observation(self) -> MatLike|dict:
        img_h, img_w, _ = self.base_img.shape

        # Calculate area so that it is always at least 1x1 pixels in a valid spot
        x1 = int(self.view[0]*img_w)
        y1 = int(self.view[1]*img_h)
        x2 = max(int(self.view[2]*img_w), x1)
        y2 = max(int(self.view[3]*img_h), y1)
        if (x1 == x2):
            if (x2 == img_w):
                x1 -= 2
            elif (x1 == 0):
                x2 += 2
            else:
                x2 +=2
        if (y1 == y2):
            if (y2 == img_h):
                y1 -= 2
            elif (y1 == 0):
                y2 += 2
            else:
                y2 +=2

        scaled_pad_x = round((x2-x1)*self.padding)
        scaled_pad_y = round((y2-y1)*self.padding)
        pad_x, pad_y = self._padding_pixels
        
        view_cutout = self.preprocessed[pad_y+y1-scaled_pad_y:pad_y+y2+scaled_pad_y, pad_x+x1-scaled_pad_x:pad_x+x2+scaled_pad_x]
        view_scaled = self._scaling(view_cutout, self.width, self.height)

        # Convert to channel-first format. See: https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
        img = np.expand_dims(view_scaled, axis=0)

        if self._state_type == ObservationType.STATE_IMAGE_ONLY:
            return img
        
        return {"image":img, "view": np.array(self.view, dtype=np.float32)}
    
    def _scaling(self, img: MatLike, width, height):
        """Special rescaling method that ensures no lines are lost"""
        h, w = img.shape[:2]
        if width > w:
            interp_x = cv2.INTER_NEAREST
        else:
            interp_x = cv2.INTER_AREA

        img_x_scaled = cv2.resize(img, (width, h), interpolation=interp_x)
        img_x_scaled[img_x_scaled > 0] = 255

        if height > h:
            interp_y = cv2.INTER_NEAREST
        else:
            interp_y = cv2.INTER_AREA

        img_final = cv2.resize(img_x_scaled, (width, height), interpolation=interp_y)
        img_final[img_final > 0] = 255
        return img_final


    
    def calculate_reward_dense(self, rect: BoundingBox, stop: bool) -> tuple[float, bool]:

        overlaping = [x for x in self.ground_truth_labels if x.has_overlap(rect)]
        leaves = [x for x in overlaping if not any([x.fully_contains(y) for y in overlaping])]

        if (len(leaves) == 0):
            print(self.ground_truth_labels, self.view)
            # This can happen in same very rare edge cases when two lines are on top of one another
            leaves = overlaping

        leaves.sort(key=lambda x: x.tolerant_iou(rect), reverse=True)

        best_bb = leaves[0]
        max_iou = best_bb.tolerant_iou(rect)
        best_is_root = best_bb is self.ground_truth_labels[0]
        self.current_best_bb = best_bb

        if stop:
            children = [x for x in self.ground_truth_labels if best_bb.fully_contains(x)]
            for child in children:
                self.ground_truth_labels.remove(child)
            self.ground_truth_labels.remove(best_bb)
            x1, y1, x2, y2 = best_bb.get_bb_corners()
            img_h, img_w, _ = self.base_img.shape
            pad_x, pad_y = self._padding_pixels
            self.preprocessed[round(pad_y+y1*img_h):round(pad_y+y2*img_h), round(pad_x+x1*img_w):round(pad_x+x2*img_w)] = 0
            self.reward_archive.append(max_iou)
            return max_iou*3, best_is_root
            
        diff = max_iou - self.last_reward
        self.last_reward = max_iou
        return diff, False

    def step(self, action):
        width = self.view[2]-self.view[0]
        heigth = self.view[3]-self.view[1]
        if action == Action.SHRINK_LEFT:
            self.view[0] = self.view[0] + width*0.15
        elif action == Action.SHRINK_TOP:
            self.view[1] = self.view[1] + heigth*0.15
        elif action == Action.SHRINK_RIGHT:
            self.view[2] = self.view[2] - width*0.15
        elif action == Action.SHRINK_BOTTOM:
            self.view[3] = self.view[3] - heigth*0.15
        elif action == Action.SHRINK_LEFT_SMALL:
            self.view[0] = self.view[0] + width*0.025
        elif action == Action.SHRINK_TOP_SMALL:
            self.view[1] = self.view[1] + heigth*0.025
        elif action == Action.SHRINK_RIGHT_SMALL:
            self.view[2] = self.view[2] - width*0.025
        elif action == Action.SHRINK_BOTTOM_SMALL:
            self.view[3] = self.view[3] - heigth*0.025
        obs = self.get_observation()

        reward, terminated, info = 0, False, {'view':self.view}

        bb = BoundingBox(self.view, BoundingBoxType.TWO_CORNERS)

        if action == Action.STOP:
            reward, terminated = self.calculate_reward_dense(bb, True)
            self.view = [0, 0, 1, 1]
            if not terminated:
                self.last_reward = self.calculate_reward_dense(BoundingBox(self.view, BoundingBoxType.TWO_CORNERS), False)[0]
            if (self.render_mode == 'human'):
                self.render()
            return self.get_observation(), reward, terminated, False, info
        else:
            reward, terminated = self.calculate_reward_dense(bb, False)


        if (self.render_mode == 'human'):
            self.render()

        return obs, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == 'none' or self.render_mode is None:
            return
        obs = self.get_observation()
        if (self._state_type == ObservationType.STATE_IMAGE_ONLY):
            obs = obs[0]
        else:
            obs = obs["image"][0]
        img = cv2.cvtColor(self.preprocessed, cv2.COLOR_GRAY2BGR)
        img_h, img_w, _ = self.base_img.shape
        view_rect = BoundingBox(self.view, BoundingBoxType.TWO_CORNERS).get_rect(img_w, img_h)
        best_bb_rect = self.current_best_bb.get_rect(img_w, img_h)

        pad_x, pad_y = self._padding_pixels

        scaled_pad_x = round(view_rect[2]*self.padding)
        scaled_pad_y = round(view_rect[3]*self.padding)

        cv2.rectangle(img, (view_rect[0]+pad_x, view_rect[1]+pad_y, view_rect[2], view_rect[3]), (0, 0, 255), 1)
        cv2.rectangle(img, (view_rect[0]+pad_x-scaled_pad_x, view_rect[1]+pad_y-scaled_pad_y, view_rect[2]+2*scaled_pad_x, view_rect[3]+2*scaled_pad_y), (0, 255, 0), 1)
        cv2.rectangle(img, (best_bb_rect[0]+pad_x, best_bb_rect[1]+pad_y, best_bb_rect[2], best_bb_rect[3]), (255, 0, 0), 1)
        scaled_obs = cv2.resize(obs, (500, 500), interpolation=cv2.INTER_NEAREST)
        if self.render_mode == 'human':
            cv2.imshow("Observation", scaled_obs)
            cv2.imshow("Current selection", img)
            cv2.waitKey(1)
            return
        if self.render_mode == "rgb_array":
            return img
        if self.render_mode == "rgb_array_list":
            return [img, scaled_obs]
        
    def close(self):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    #env = gymnasium.make('square-v8-discrete', render_mode='none', height = 84, width = 84, start_rects = 1000)

    print("check begin")
    #check_env(env)
    print("check end")

    """total = 0
    for _ in range(10000):
        env.reset()
        #print(env.env.env.__dict__)
        total += env.env.env.steps

    print(total/10000)"""

    dataset = r"rl\dataset_big"

    image_paths = [join(dataset, f) for f in listdir(dataset) if isfile(join(dataset, f))][1000:2000]
    total = 0
    for i in image_paths:
        base_img = cv2.imread(i)
        total += len(find_bounding_boxes(base_img))

    print(total/len(image_paths))
    exit(0)


    obs = env.reset(seed=1)[0]

    for i in range(1000):
        rand_action = env.action_space.sample()
        print(rand_action)
        obs, reward, terminated, _, _ = env.step(rand_action)
        print(reward, terminated)
        if (terminated):
            cv2.waitKey(0)
            obs = env.reset()[0]