from typing import Literal

import numpy as np

__all__ = (
    'CoordinateArray',
    'CoordinateArray2D',
    'CoordinateArray3D',
    'EdgeArray',
    'TriangleArray',
)

CoordinateArray2D = np.ndarray[tuple[int, Literal[2]], np.dtype[np.float32]]
CoordinateArray3D = np.ndarray[tuple[int, Literal[3]], np.dtype[np.float32]]
CoordinateArray = CoordinateArray2D | CoordinateArray3D
EdgeArray = np.ndarray[tuple[int, Literal[2]], np.dtype[np.int64]]
TriangleArray = np.ndarray[tuple[int, Literal[3]], np.dtype[np.int32]]
BoxArray = np.ndarray[tuple[Literal[9], int], np.dtype[np.float32]]
