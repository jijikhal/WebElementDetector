import random
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
from numpy.typing import NDArray
from typing import cast

register(
    id='square-v9-discrete',
    entry_point='square_v9_env_discrete:SquareEnv'
)

SHRINK_LEFT = 0
SHRINK_RIGHT = 1
SHRINK_TOP = 2
SHRINK_BOTTOM = 3
SHRINK_LEFT_SMALL = 4
SHRINK_RIGHT_SMALL = 5
SHRINK_TOP_SMALL = 6
SHRINK_BOTTOM_SMALL = 7
STOP = 8

STATE_IMAGE_ONLY = 0
STATE_IMAGE_AND_VIEW = 1

def draw_rect(bb: RectI, image: MatLike) -> None:
    x, y, w, h = bb
    cv2.rectangle(image, (x, y), (x+w-1, y+h-1), (255,), 1)

def find_bounding_boxes(img: MatLike, max_bbs: int = 1000) -> tuple[dict[BoundingBox, int], MatLike, MatLike]:
    copy_for_show = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.bilateralFilter(gray, 21, 50, 0)
    binary = cv2.Canny(smoothed, 22, 22, L2gradient=True)
    big_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    big_count, labels, big_stats, _ = cv2.connectedComponentsWithStats(big_close, connectivity=8)
    img_h, img_w, _ = img.shape

    found_components: list[tuple[BoundingBox, int]] = [(BoundingBox((0, 0, 1, 1), BoundingBoxType.TWO_CORNERS), big_count)]
    cv2.rectangle(labels, (0, 0, img_w-1, img_h-1), (big_count,), 1)

    for i in range(1, big_count):
        x, y, w, h, area = big_stats[i]
        if (min(w,h) < 10 or max(w, h) < 20):
            cv2.rectangle(copy_for_show, (x, y), (x + w + 1, y + h + 1), (0, 0, 255), 1)
            continue
        found_components.append((BoundingBox((x/img_w, y/img_h, (w+1)/img_w, (h+1)/img_h), BoundingBoxType.TOP_LEFT), i))
        cv2.rectangle(copy_for_show, (x, y), (x + w + 1, y + h + 1), (0, 255, 0), 2)

    merged: list[tuple[BoundingBox, list[int]]] = []
    used = set()
    found_components.sort(key=lambda x: x[0].area())
    for b, i in found_components:
        if (b in used):
            continue
        used.add(b)
        to_merge_with = [(x, j) for x, j in found_components if b.is_intersecting(x) and x not in used]
        merged_labels = [i]
        m = b
        for t, j in to_merge_with:
            used.add(t)
            merged_labels.append(j)
            m = b.merge(t)
        merged.append((m, merged_labels))

    for b, _ in merged:
        x1, y1, x2, y2 = b.get_bb_corners()
        cv2.rectangle(copy_for_show, (round(x1*img_w), round(y1*img_h)), (round(x2*img_w), round(y2*img_h)), (255, 0, 0), 2)

    merged.sort(key=lambda x: x[0].area(), reverse=True)

    result = np.zeros_like(labels, dtype=np.uint8)
    label_map = np.zeros_like(labels)
    box_label_mapping: dict[BoundingBox, int] = {}

    for i in range(min(len(merged), max_bbs)):
        box, corresponding_labels = merged[i]
        mapped_label = i+1
        box_label_mapping[box] = mapped_label
        for l in corresponding_labels:
            result[labels == l] = 255
            label_map[labels == l] = mapped_label

    #cv2.imshow("img", copy_for_show)
    #cv2.imshow("img2", result)
    #cv2.waitKey(0)
    
    return box_label_mapping, label_map, result

class SquareEnv(gymnasium.Env):
    metadata = {'render_modes': ['human','none', 'rgb_array_list', 'rgb_array']} 
    def __init__(self, height: int = 100, width: int = 100, render_mode=None, dataset_folder: str = "dataset_big", start_rects: int = 3, name: str = "env", state_type = STATE_IMAGE_ONLY, padding: float = 0.00) -> None:
        super().__init__()
        self.height: int = height
        self.width: int = width
        self.render_mode = render_mode
        self.image_paths = [join(dataset_folder, f) for f in listdir(dataset_folder) if isfile(join(dataset_folder, f))]
        self.base_img_path = random.choice(self.image_paths)
        self.base_img = cv2.imread(self.base_img_path)
        self.max_bbs = start_rects
        box_label_mapping, label_map, result = find_bounding_boxes(self.base_img, self.max_bbs)
        self.ground_truth_labels: dict[BoundingBox, int] = box_label_mapping
        self.label_map = label_map
        self.view: list[float] = [0.0, 0.0, 1.0, 1.0]
        self.last_reward = 0
        self.padding = padding
        self._padding_pixels: tuple[int, int] = (0,0)
        self.preprocessed: MatLike = self._preprocess_img(result)
        self.reward_archive: list[float] = []
        self.current_best_bb = BoundingBox((0, 0, 1, 1))
        self.name = name
        self._state_type = state_type

        self.action_space = spaces.Discrete(9)
        if (self._state_type == STATE_IMAGE_ONLY):
            self.observation_space = spaces.Box(low=0, high=255, shape=(1, height, width), dtype=np.uint8)
        elif (self._state_type == STATE_IMAGE_AND_VIEW):
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
            if (avg > 0.8):
                self.max_bbs += 1
                print(f"Increased difficulty of env {self.name} to {self.max_bbs}")
                try:
                    with open("difficulty_log.txt", "a") as f:
                        f.write(f"Increased difficulty of env {self.name} to {self.max_bbs} at {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}\n")
                except:
                    pass

            self.reward_archive = []


        box_label_mapping, label_map, result = find_bounding_boxes(self.base_img, self.max_bbs)
        self.ground_truth_labels = box_label_mapping
        self.label_map = label_map
        self.preprocessed = self._preprocess_img(result)

        self.last_reward = 0
        self.last_reward = self.calculate_reward_dense(BoundingBox(self.view, BoundingBoxType.TWO_CORNERS), False)[0]

        obs = self.get_observation()
        info = {}

        if (self.render_mode == 'human'):
            self.render()

        return obs, info
    
    def _preprocess_img(self, img: MatLike) -> MatLike:
        img_h, img_w, _ = self.base_img.shape

        padding_x = round(img_w*self.padding)
        padding_y = round(img_h*self.padding)
        self._padding_pixels = (padding_x, padding_y)

        padded = np.pad(img, ((padding_y, padding_y),(padding_x, padding_x)), 'constant', constant_values=0)
        return padded
    
    def get_observation(self) -> MatLike|dict[str, NDArray[np.uint8|np.float32]]:
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

        if self._state_type == STATE_IMAGE_ONLY:
            return img
        
        return {"image":img, "view": np.array(self.view, dtype=np.float32)}
    
    def _scaling(self, img: MatLike, width, height):
        """Special rescaling method that ensures no lines are lost"""
        h, w = img.shape[:2]
        if width > w:
            interp_x = cv2.INTER_NEAREST
        else:
            interp_x = cv2.INTER_AREA

        img_x_scaled = cast(NDArray[np.uint8], cv2.resize(img, (width, h), interpolation=interp_x))
        img_x_scaled[img_x_scaled > 0] = 255

        if height > h:
            interp_y = cv2.INTER_NEAREST
        else:
            interp_y = cv2.INTER_AREA

        img_final = cast(NDArray[np.uint8], cv2.resize(img_x_scaled, (width, height), interpolation=interp_y))
        img_final[img_final > 0] = 255
        return img_final


    
    def calculate_reward_dense(self, rect: BoundingBox, stop: bool) -> tuple[float, bool]:

        boxes = list(sorted(self.ground_truth_labels.keys(), key=lambda x: x.area(), reverse=True))
        overlaping = [x for x in boxes if x.has_overlap(rect)]
        leaves = [x for x in overlaping if not any([x.fully_contains(y) for y in overlaping])]

        if (len(leaves) == 0):
            print(self.ground_truth_labels, self.view)
            # This can happen in same very rare edge cases when two lines are on top of one another
            leaves = overlaping

        leaves.sort(key=lambda x: x.iou(rect), reverse=True)

        best_bb = leaves[0]
        max_iou = best_bb.iou(rect)
        best_is_root = best_bb is boxes[0]
        self.current_best_bb = best_bb

        if stop:
            children = [x for x in self.ground_truth_labels if best_bb.fully_contains(x)]
            to_erase_labels: list[int] = []
            for child in children:
                to_erase_labels.append(self.ground_truth_labels.pop(child))
            to_erase_labels.append(self.ground_truth_labels.pop(best_bb))
            pad_x, pad_y = self._padding_pixels
            for l in to_erase_labels:
                if (pad_x > 0):
                    self.preprocessed[pad_y:-pad_y, pad_x:-pad_x][self.label_map == l] = 0
                else:
                    self.preprocessed[self.label_map == l] = 0
            self.reward_archive.append(max_iou)
            return max_iou*3, best_is_root
            
        diff = max_iou - self.last_reward
        self.last_reward = max_iou
        return diff, False

    def step(self, action):
        width = self.view[2]-self.view[0]
        heigth = self.view[3]-self.view[1]
        if action == SHRINK_LEFT:
            self.view[0] = self.view[0] + width*0.15
        elif action == SHRINK_TOP:
            self.view[1] = self.view[1] + heigth*0.15
        elif action == SHRINK_RIGHT:
            self.view[2] = self.view[2] - width*0.15
        elif action == SHRINK_BOTTOM:
            self.view[3] = self.view[3] - heigth*0.15
        elif action == SHRINK_LEFT_SMALL:
            self.view[0] = self.view[0] + width*0.025
        elif action == SHRINK_TOP_SMALL:
            self.view[1] = self.view[1] + heigth*0.025
        elif action == SHRINK_RIGHT_SMALL:
            self.view[2] = self.view[2] - width*0.025
        elif action == SHRINK_BOTTOM_SMALL:
            self.view[3] = self.view[3] - heigth*0.025
        obs = self.get_observation()

        reward, terminated, info = 0, False, {'view':self.view}

        bb = BoundingBox(self.view, BoundingBoxType.TWO_CORNERS)

        if action == STOP:
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
        if (self._state_type == STATE_IMAGE_ONLY):
            obs = cast(NDArray[np.uint8], self.get_observation())
            obs_img = obs[0]
        else:
            obs = cast(dict[str, NDArray[np.uint8]], self.get_observation())
            obs_img = obs["image"][0]
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
        scaled_obs = cv2.resize(obs_img, (500, 500), interpolation=cv2.INTER_NEAREST)
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
    env = gymnasium.make('square-v9-discrete', render_mode='none', height = 84, width = 84, start_rects = 1000)

    print("check begin")
    check_env(env)
    print("check end")

    """total = 0
    for _ in range(10000):
        env.reset()
        #print(env.env.env.__dict__)
        total += env.env.env.steps

    print(total/10000)"""


    obs = env.reset(seed=1)[0]

    for i in range(1000):
        rand_action = env.action_space.sample()
        print(rand_action)
        obs, reward, terminated, _, _ = env.step(rand_action)
        print(reward, terminated)
        if (terminated):
            cv2.waitKey(0)
            obs = env.reset()[0]