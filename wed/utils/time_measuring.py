from wed.utils.predict import Detector, YoloDetector, CVDetector, RLDetector
from wed.utils.get_files_in_folder import get_files
from typing import Callable
from time import perf_counter
import random
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

def run_test(detector_func: Callable[[], Detector], data_folder: str, run_count: int, predictions_per_run: int, seed: int|None = None):
    init_times: list[float] = []
    first_run_times: list[float] = []
    predict_run_times: dict[int, list[float]] = {}

    random.seed(seed)
    files = get_files(data_folder)
    get_rand_image = lambda: cv2.imread(files[random.randint(0, len(files)-1)])

    for i in range(run_count):
        print(f"Starting run {i}")
        #print("Inicializing detector")
        start = perf_counter()
        detector = detector_func()
        end = perf_counter()
        init_time = end-start
        #print(f"Inicialization took {init_time/1e6:.2f} ms")
        init_times.append(init_time*1000)

        img = get_rand_image()
        start = perf_counter()
        result = detector.predict(img)
        end = perf_counter()
        first_predict_time = end-start

        first_run_times.append(first_predict_time*1000)

        for _ in range(predictions_per_run):
            img = get_rand_image()
            start = perf_counter()
            result = detector.predict(img)
            end = perf_counter()
            predict_time = end-start

            if not len(result) in predict_run_times:
                predict_run_times[len(result)] = []
            predict_run_times[len(result)].append(predict_time*1000)

    print(init_times)
    print(first_run_times)

    print("median", np.percentile(init_times, 50))
    print("median", np.percentile(first_run_times, 50))
    #print(predict_run_times)

    plot_time_per_count(predict_run_times)

def plot_time_per_count(data: dict[int, list[float]]):
    bin_edges = list(range(0, 101, 10))  # Bins: 0–10, 11–20, ..., 91–100
    bin_labels = [f"{i+1}-{i+10}" for i in range(0, 100, 10)]

    # Collect flattened bin labels and timings
    group_labels = []
    timings = []

    for object_count, time_list in data.items():
        # Find which bin this object_count falls into
        bin_index = np.digitize(object_count, bin_edges, right=True) - 1
        if 0 <= bin_index < len(bin_labels):
            bin_label = bin_labels[bin_index]
            group_labels.extend([bin_label] * len(time_list))
            timings.extend(time_list)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=group_labels, y=timings, order=bin_labels)
    plt.xlabel("Object Count Range")
    plt.ylabel("Prediction Time (ms)")
    plt.title("CV-detector prediction times by object count")
    plt.xticks(rotation=45)
    ymax = np.percentile(timings, 99)*1.2
    ymin = np.percentile(timings, 1)*0.8
    print("median", np.percentile(timings, 50))
    plt.ylim(ymin, ymax)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    r"""run_test(
        lambda: YoloDetector(r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\wed\yolo\runs\detect\train5\weights\best.pt"),
        r"yolo\dataset\images\test",
        100,
        1,
        1
    )"""

    run_test(
        lambda: CVDetector(),
        r"yolo\dataset\images\test",
        1,
        100,
        1
    )

