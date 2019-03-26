from enum import Enum


class Mode(Enum):
    PAN_ZOOM = 0
    SELECT = 1
    DIRECT = 2
    ADD_RECTANGLE = 3
    ADD_ELLPISE = 4
    ADD_LINE = 5
    ADD_PATH = 6
    ADD_POLYGON = 7
    VERTEX_INSERT = 8
    VERTEX_REMOVE = 9


BOX_WITH_HANDLE = [0, 1, 2, 3, 4, 5, 6, 7, 9]
BOX_LINE_HANDLE = [1, 2, 4, 6, 0, 1, 8]
BOX_LINE = [0, 2, 4, 6, 0]
BOX_TOP = 1
BOX_HANDLE = 9
BOX_CENTER = 8
BOX_LEN = 8
