from enum import Enum


class InteractionBoxHandle(int, Enum):
    # vertices are generated according to the following scheme:
    # TODO: outdated
    #      8
    #      |
    #  1---4---3
    #  |       |
    #  5       7
    #  |       |
    #  0---6---2
    TOP_LEFT = 1
    TOP_CENTER = 4
    TOP_RIGHT = 3
    CENTER_LEFT = 5
    CENTER_RIGHT = 7
    BOTTOM_LEFT = 0
    BOTTOM_CENTER = 6
    BOTTOM_RIGHT = 2
    ROTATION = 8
