# WebElementDetector

This repository contains the source code used in my bachelor's thesis. That experimented with the use of deep reinforcement learning for grephical user interface element detection.

## Structure

All the source code is located in the `wed` directory. It contains four more important folders:

- `cv` -- contains all source code of the two detectors based on traditional computer vision algorithms
- `rl` -- contains all the RL environments and scripts for training and testing the RL agents
- `yolo` -- contains the scripts for dataset creation and training of the YOLOv8 based detector
- `utils` -- contains some scripts common for all detectors. Namely for running them and some other common things.
- `thesis_images` -- scripts for generating images in the thesis text.

Each directory also has its own `README.md` describing contents of that directory.

## Running

Everything was developed and tested using Python 3.11.1, although any Python 3.10+ should likely work (the newest used feature is the `|` operator in type annotations).

The required libraries are specified in `wed/requirements.txt` and can be installed using `pip install -r wed/requirements.txt`. For training, it is recommended to use CUDA if available. For that, install torch using `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126`.

OpenCV also requires some further things. On Ubuntu run `sudo apt install ffmpeg libsm6 libxext6` and OpenCV should work. If it still throws "Module not found" please use the OpenCV documentation to solve the issues.

The `wed` folder is a Python module and most of the scripts rely on being able to import it. You will likely have to add the directory containing the `wed` directory into the `PYTHONPATH` env variable (on both Linux and Windows).

Some of the scripts may reference models and dataset files not included in the repository. The used dataset can be downloaded at https://drive.google.com/drive/folders/1cNjcwM0rc8Wr9dESrsxytKwXl9j3QnbR.