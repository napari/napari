from collections.abc import Callable

import numpy as np
from numpy import typing as npt

from napari.layers._scalar_field._scalar_field_constants import (
    ImageProjectionMode,
)


def project_slice(
    data: npt.NDArray, axis: tuple[int, ...], mode: ImageProjectionMode
) -> npt.NDArray:
    """Project a thick slice along axis based on mode."""
    if all(data.shape[axis] == 1 for axis in axis):
        # If all axes are of size 1, return the data as is
        return data[
            tuple(
                slice(None) if i not in axis else 0 for i in range(data.ndim)
            )
        ]
    func: Callable
    if mode == ImageProjectionMode.SUM:
        func = np.sum
    elif mode == ImageProjectionMode.MEAN:
        func = np.mean
    elif mode == ImageProjectionMode.MAX:
        func = np.max
    elif mode == ImageProjectionMode.MIN:
        func = np.min
    else:
        raise NotImplementedError(f'unimplemented projection: {mode}')
    return func(data, tuple(axis))
