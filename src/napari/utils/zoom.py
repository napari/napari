from __future__ import annotations

import typing as ty
import warnings
from collections.abc import Iterable

import numpy as np


def find_nearest_index(data: np.ndarray, value: float | np.ndarray | Iterable):
    """Find nearest index of asked value.

    Parameters
    ----------
    data : np.array
        input array (e.g. m/z values)
    value : Union[int, float, np.ndarray]
        asked value

    Returns
    -------
    index :
        index value
    """
    data = np.asarray(data)
    if isinstance(value, Iterable):
        return np.asarray(
            [np.argmin(np.abs(data - _value)) for _value in value],
            dtype=np.int64,
        )
    return np.argmin(np.abs(data - value))


def find_nearest_value(
    data: ty.Iterable, value: float | np.ndarray | Iterable
):
    """Find nearest value."""
    data = np.asarray(data)
    idx = find_nearest_index(data, value)
    return data[idx]


def get_extents_from_layers(viewer) -> tuple[float, float, float, float]:
    """Calculate extents from all layers."""
    extents_ = []
    for layer in viewer.layers:
        if not hasattr(layer, '_extent_data'):
            continue
        ext = layer._extent_data
        if np.isnan(ext).all():
            continue
        mins = np.min(layer._extent_data, axis=0)
        maxs = np.max(layer._extent_data, axis=0)
        extents_.append((mins[0], maxs[0], mins[1], maxs[1]))
    if not extents_:
        extents_ = [(0, 512, 0, 512)]
    extents = np.asarray(extents_)
    if np.all(np.isnan(extents)):
        return 0, 512, 0, 512
    return (
        np.nanmin(extents[:, 0]),
        np.nanmax(extents[:, 1]),
        np.nanmin(extents[:, 2]),
        np.nanmax(extents[:, 3]),
    )


def get_multiplier(xmax: float, ymax: float) -> float:
    """Based on the maximum value, get a multiplier."""
    max_size = max(xmax, ymax)
    range_to_multiplier = {
        1_000: 0.75,
        2_500: 0.5,
        5_000: 0.35,
        10_000: 0.25,
        15_000: 0.05,
        23_000: 0.03,
        100_000: 0.01,
        250_000: 0.005,
        float('inf'): 0.005,
    }
    nearest = find_nearest_value(list(range_to_multiplier.keys()), max_size)
    return range_to_multiplier[nearest]


def calculate_zoom(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    viewer,
    multiplier: float | None = None,
) -> tuple[float, float, float]:
    """Calculate zoom for specified region."""
    # calculate min/max for y, x coordinates
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    # calculate extents for the view
    x_min_, x_max_, y_min_, y_max_ = get_extents_from_layers(viewer)
    if multiplier is None:
        multiplier = get_multiplier(x_max_, y_max_)

    # calculate zoom as fraction of the extent
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        if y_max_ > x_max_:
            zoom = ((y_max_ - y_min_) / (y_max - y_min)) * multiplier
        else:
            zoom = ((x_max_ - x_min_) / (x_max - x_min)) * multiplier
    if np.isinf(zoom) or np.isnan(zoom) or zoom == 0:
        zoom = 1
    return zoom, y_center, x_center
