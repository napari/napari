"""Vispy zoom box overlay."""

from __future__ import annotations

import numpy as np

from napari.viewer import ViewerModel


def calculate_zoom_proportion(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    viewer: ViewerModel,
) -> tuple[float, float, float]:
    """Calculate zoom for specified region."""
    # calculate the center of the rectangle
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    # using the viewer's scene size to calculate zoom
    _, scene_size, _, _ = viewer._get_scene_parameters()
    y_diff = y_max - y_min
    x_diff = x_max - x_min
    # calculate average zoom based on the size of the rectangle
    zoom = np.mean(scene_size / (y_diff, x_diff))
    # ensure zoom is a valid number
    if np.isinf(zoom) or np.isnan(zoom) or zoom == 0:
        zoom = 1
    return zoom, y_center, x_center
