from functools import lru_cache
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

import numpy as np
from skimage import morphology


class Skiper:
    def __init__(self, func):
        self.func = func

    def __contains__(self, item):
        return self.func(item)


def _generate_ball(radius: int, ndim: int) -> np.ndarray:
    """Generate a ball of given radius and dimension.

    Parameters
    ----------
    radius : int
        Radius of the ball.
    ndim : int
        Dimension of the ball.

    Returns
    -------
    ball : ndarray of uint8
        Binary array of the hyper ball.
    """

    if ndim == 2:
        return morphology.disk(radius)
    if ndim == 3:
        return morphology.ball(radius)
    shape = (2 * radius + 1,) * ndim
    radius_sq = radius**2
    coords = np.indices(shape) - radius
    return (np.sum(coords**2, axis=0) <= radius_sq).astype(np.uint8)


def _generate_density(radius: int, ndim: int) -> np.ndarray:
    """Generate gaussian density of given radius and dimension."""
    shape = (2 * radius + 1,) * ndim
    coords = np.indices(shape) - radius
    dist = np.sqrt(np.sum(coords**2 / ((radius / 4) ** 2), axis=0))
    res = np.exp(-dist)
    res[res < 0.02] = 0
    return res


def _add_structure_on_coordinates(
    data: np.ndarray,
    points: np.ndarray,
    structure: np.ndarray,
    values: Sequence,
    assign_operator: Callable[[np.ndarray, np.ndarray, Any], np.ndarray],
):
    """Update data with structure at given coordinates.

    Parameters
    ----------
    data : ndarray
        Array to update.
    points : ndarray
        Coordinates of the points. The structures will be added at these
        points (center).
    structure : ndarray
        Array with encoded structure. For example, ball (boolean) or density
        (0,1) float.
    values : ndarray
        Values to assign to the structure. It is passed to the assign_operator.
        Could be used for labeling.
    assign_operator : function
        Function to assign structure to the data. It takes clipped data,
        structure and value as arguments.
    """
    radius = (structure.shape[0] - 1) // 2
    shape = data.shape

    for j, point in enumerate(points.T):
        slice_im = []
        slice_ball = []
        for i, p in enumerate(point):
            slice_im.append(
                slice(max(0, p - radius), min(shape[i], p + radius + 1))
            )
            ball_base = max(0, radius - p)
            bal_end = slice_im[-1].stop - slice_im[-1].start + ball_base
            slice_ball.append(slice(ball_base, bal_end))
        assign_operator(
            data[tuple(slice_im)], structure[tuple(slice_ball)], values[j]
        )
    return data


def _update_data_with_mask(data, struct, value):
    """Helper function to generate labeling array"""
    data[struct > 0] = value
    return data


def _add_value_to_data(data, struct, _value):
    """Helper function to generate nice density array"""
    data[...] = np.max([data, struct], axis=0)
    return data


@overload
def labeled_particles(
    shape: Sequence[int],
    dtype: Optional[np.dtype] = None,
    n: int = 144,
    seed: Optional[int] = None,
    return_density: Literal[False] = False,
) -> np.ndarray:
    ...


@overload
def labeled_particles(
    shape: Sequence[int],
    dtype: Optional[np.dtype] = None,
    n: int = 144,
    seed: Optional[int] = None,
    return_density: Literal[True] = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...


@lru_cache
def labeled_particles(
    shape: Sequence[int],
    dtype: Optional[np.dtype] = None,
    n: int = 144,
    seed: Optional[int] = None,
    return_density: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate labeled blobs of given shape and dtype.

    Parameters
    ----------
    shape : Sequence[int]
        Shape of the resulting array.
    dtype : np.dtype
        Dtype of the resulting array.
    n : int
        Number of blobs to generate.
    seed : Optional[int]
        Seed for the random number generator.
    return_density : bool
        Whether to return the density array and center coordinates.
    """
    if dtype is None:
        for dtype_ in [np.uint8, np.uint16, np.uint32, np.uint64]:
            if np.iinfo(dtype_).max >= n:
                dtype = dtype_
                break
        else:
            raise ValueError(f"n is too large for any dtype: {n=}")
    rng = np.random.default_rng(seed)
    points = (
        rng.random((len(shape), n)) * np.array(shape).reshape((-1, 1))
    ).astype(int)
    values = rng.integers(
        np.iinfo(dtype).min, np.iinfo(dtype).max, size=n, dtype=dtype
    )
    sigma = int(np.array(shape).max() / (4.0 * n ** (1 / len(shape))))
    ball = _generate_ball(sigma, len(shape))
    balls_ = np.zeros(shape, dtype=dtype)

    _add_structure_on_coordinates(
        balls_, points, ball, values, _update_data_with_mask
    )

    if return_density:
        particles = np.zeros(shape, dtype=np.float32)
        dens = _generate_density(sigma * 2, len(shape))
        _add_structure_on_coordinates(
            particles, points, dens, values, _add_value_to_data
        )

        return balls_, particles, points.T
    return balls_
