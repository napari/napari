from typing import Literal

import numpy as np

__all__ = (
    'CoordinateArray',
    'CoordinateArray2D',
    'CoordinateArray3D',
    'EdgeArray',
    'TriangleArray',
)

CoordinateArray = np.ndarray[tuple[int, Literal[2, 3], np.float32]]
CoordinateArray2D = np.ndarray[tuple[int, Literal[2]], np.float32]
CoordinateArray3D = np.ndarray[tuple[int, Literal[3]], np.float32]
EdgeArray = np.ndarray[tuple[int, Literal[2]], np.int64]
TriangleArray = np.ndarray[tuple[int, Literal[3]], np.int32]
