# This file contains script for generating dataset for the YOLO model. Used in Section 5.2. The YAML file must be created manually.
from os import listdir, makedirs
from os.path import isfile, join
import cv2
from wed.cv.detector import find_elements_cv
from tqdm import tqdm
from shutil import copy
from random import seed, shuffle


def create_labels(images: str, labels: str) -> None:
    """Generates labels for provided images in the YOLOv8 format

    Args:
        images (str): path to folder with images to be labeled
        labels (str): path to folder where labels are to be stored
    """
    makedirs(labels, exist_ok=True)

    image_files = listdir(images)
    for i in tqdm(image_files, desc="Labeling images in dataset"):
        img_path = join(images, i)
        if not (isfile(img_path)):
            print(f"Failed to find image {i}")
            continue
        label_path = join(labels, ".".join(i.split(".")[:-1])) + ".txt"

        img = cv2.imread(img_path)
        try:
            boxes, _ = find_elements_cv(img, include_root=False)
        except Exception as e:
            print(f"Failed to annotate image {i}: {e}")
            continue
        with open(label_path, "w") as f:
            for b in boxes:
                x, y, w, h = b.get_bb_middle()
                f.write(f"0 {x:6f} {y:6f} {w:6f} {h:6f}\n")


def train_test_val_split(image_folder: str, result_folder: str, ratio: tuple[int, int, int], shuffle_seed: int | None = None, max_train_size: int | None = None):
    """Splits the provided data into train/val/test and generates labels

    Args:
        image_folder (str): path to folder with all images
        result_folder (str): path to folder where folders for labels and images will be created
        ratio (tuple[int, int, int]): The ratio of train:val:test. Recommended is (8, 1, 1)
        shuffle_seed (int | None, optional): Seed to use when shuffeling the data. Defaults to None.
        max_train_size (int | None, optional): Largest allowed size of train dataset. Defaults to None.
    """
    image_files = listdir(image_folder)
    tr, v, te = ratio
    total_ratio = tr+v+te
    v_count = round(v/total_ratio*len(image_files))
    te_count = round(te/total_ratio*len(image_files))
    tr_count = len(image_files)-v_count-te_count
    if max_train_size is not None:
        tr_count = min(tr_count, max_train_size)

    seed(shuffle_seed)
    shuffle(image_files)

    img_folder = join(result_folder, "images")
    labels_folder = join(result_folder, "labels")

    tr_img_path = join(img_folder, "train")
    v_img_path = join(img_folder, "val")
    te_img_path = join(img_folder, "test")

    makedirs(tr_img_path, exist_ok=True)
    makedirs(v_img_path, exist_ok=True)
    makedirs(te_img_path, exist_ok=True)

    tr_lab_path = join(labels_folder, "train")
    v_lab_path = join(labels_folder, "val")
    te_lab_path = join(labels_folder, "test")

    for i in tqdm(image_files[:tr_count], desc="Copying training data"):
        try:
            copy(join(image_folder, i), join(tr_img_path, i))
        except Exception as e:
            print(f"Error when copying image {i}: {e}")

    for i in tqdm(image_files[tr_count:tr_count+v_count], desc="Copying validation data"):
        try:
            copy(join(image_folder, i), join(v_img_path, i))
        except Exception as e:
            print(f"Error when copying image {i}: {e}")

    for i in tqdm(image_files[tr_count+v_count:tr_count+v_count+te_count], desc="Copying test data"):
        try:
            copy(join(image_folder, i), join(te_img_path, i))
        except Exception as e:
            print(f"Error when copying image {i}: {e}")

    create_labels(tr_img_path, tr_lab_path)
    create_labels(v_img_path, v_lab_path)
    create_labels(te_img_path, te_lab_path)


if __name__ == "__main__":
    IMAGE_FOLDER = r"rl\dataset_big"
    RESULT_FOLDER = r"yolo\dataset_100"
    train_test_val_split(IMAGE_FOLDER, RESULT_FOLDER, (8, 1, 1), 0, 100)
