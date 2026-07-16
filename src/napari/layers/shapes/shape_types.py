from typing import Literal

import numpy as np

__all__ = (
    'BoxArray',
    'CoordinateArray',
    'CoordinateArray2D',
    'CoordinateArray3D',
    'CoordinateDtype',
    'EdgeArray',
    'IndexArray',
    'IndexDtype',
    'ShapeColor',
    'ShapeColorArray',
    'ShapeColorDtype',
    'TriangleArray',
    'TriangleDtype',
    'ZOrderArray',
    'ZOrderDtype',
)


CoordinateDtype = np.float32
IndexDtype = np.int32
ShapeColorDtype = np.float32
TriangleDtype = np.int32
ZOrderDtype = np.int32

BoxArray = np.ndarray[tuple[Literal[9], int], np.dtype[np.float32]]
CoordinateArray2D = np.ndarray[
    tuple[int, Literal[2]], np.dtype[CoordinateDtype]
]
CoordinateArray3D = np.ndarray[
    tuple[int, Literal[3]], np.dtype[CoordinateDtype]
]
CoordinateArray = CoordinateArray2D | CoordinateArray3D
EdgeArray = np.ndarray[tuple[int, Literal[2]], np.dtype[np.int64]]
IndexArray = np.ndarray[tuple[int], np.dtype[IndexDtype]]
ShapeColor = np.ndarray[tuple[Literal[4]], np.dtype[ShapeColorDtype]]
ShapeColorArray = np.ndarray[tuple[int, Literal[4]], np.dtype[ShapeColorDtype]]
TriangleArray = np.ndarray[tuple[int, Literal[3]], np.dtype[TriangleDtype]]
ZOrderArray = np.ndarray[tuple[int], np.dtype[ZOrderDtype]]
