from enum import Enum


class InteractionBoxHandle(int, Enum):
    # vertices are generated according to the following scheme:
    # (y is actually upside down in the canvas)
    #      8
    #      |
    #  0---4---2    1 = position
    #  |       |
    #  5   9   6
    #  |       |
    #  1---7---3
    TOP_LEFT = 0
    TOP_CENTER = 4
    TOP_RIGHT = 2
    CENTER_LEFT = 5
    CENTER_RIGHT = 6
    BOTTOM_LEFT = 1
    BOTTOM_CENTER = 7
    BOTTOM_RIGHT = 3
    ROTATION = 8
    ALL = 9
