from enum import IntEnum

class Action(IntEnum):
    SHRINK_LEFT = 0
    SHRINK_RIGHT = 1
    SHRINK_TOP = 2
    SHRINK_BOTTOM = 3
    SHRINK_LEFT_SMALL = 4
    SHRINK_RIGHT_SMALL = 5
    SHRINK_TOP_SMALL = 6
    SHRINK_BOTTOM_SMALL = 7
    STOP = 8

class ObservationType(IntEnum):
    STATE_IMAGE_ONLY = 0
    STATE_IMAGE_AND_VIEW = 1