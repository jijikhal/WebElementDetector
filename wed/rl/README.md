# Reinforcement learning based detector

This directory contains all the environments described in Chapter 4 and a bunch of
small scripts for training and testing.

## Directory envs

Contains all the custom environments.

- `common.py` -- contains enums common to multiple environments.
- `image_generator.py` -- contains a generator of hierarchical environment states (Section 4.11)
- `square_v0_env.py` -- environment used in Section 4.4 containing one static square
- `square_v1_env.py` -- environment used in Section 4.5 containing randomly placed square
- `square_v2_env.py` -- environment used in Section 4.6 containing one randomly placed rectangle
- `square_v2_env_discrete.py` -- environment used in Section 4.13 containing one randomly placed rectangle with discrete actions
- `square_v3_env.py` -- environment used in Section 4.7 containing multiple randomly placed rectangles
- `square_v3_env_discrete.py` -- environment used in Section 4.14 containing multiple randomly placed rectangles with discrete actions
- `square_v4_env.py` -- environment used in Section 4.8 containing multiple randomly placed shapes
- `square_v5_env.py` -- environment used in Section 4.9 containing multiple randomly placed shapes but only outlines
- `square_v6_env.py` -- environment used in Section 4.10 containing multiple randomly placed shapes, only outlines, bigger images
- `square_v7_env.py` -- environment used in Section 4.11 containing hierarchicaly placed rectangles
- `square_v7_env_discrete.py` -- environment used in Section 4.15 containing hierarchicaly placed rectangles with discrete actions
- `square_v8_env_discrete.py` -- environment used in Section 4.16 containing hierarchicaly placed rectangles based on real screenshots with discrete actions
- `square_v9_env_discrete.py` -- environment used in Section 4.17 containing objects from real images with discrete actions

## Directory nn

Contains feature extractors and other things related to training

- `bigger_net_feature_extractor.py` -- feature extractor used in Subsection 4.7.3
- `combined_feature_extractor.py` -- feature extractor used in Subsection 4.16.4
- `resnet_combined_feature_extractor.py` -- feature extractor mentioned in Subsection 4.17.1
- `resnet_feature_extractor.py` -- feature extractor used in Subsection 4.8.1
- `squeezenet_combined_feature_extractor.py` -- feature extractor used in Subsection 4.17.1

- `schedules.py` -- definitions of schedules for hyperparameter decaying

## Other files

- `discrete_env_debug.py` -- script for debugging environments with discrete actions. Allows user to make actions with arrows, WASD and spacebar
- `evaluate.py` -- script for runnung trained RL agents and viewing their actions
- `optimize_hyperparameters` -- script for finding hyperparameters with Optuna
- `train.py` -- script for running trainings of RL agents