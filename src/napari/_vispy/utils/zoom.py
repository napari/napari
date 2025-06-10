"""Vispy zoom box overlay."""

from __future__ import annotations

import numpy as np

from napari.viewer import ViewerModel


def calculate_zoom_proportion(
    dim1_min: float,
    dim1_max: float,
    dim2_min: float,
    dim2_max: float,
    viewer: ViewerModel,
) -> tuple[float, float, float]:
    """Calculate zoom for specified region."""
    # calculate the center of the rectangle
    dim1_center = (dim1_min + dim1_max) / 2
    dim2_center = (dim2_min + dim2_max) / 2
    # using the viewer's scene size to calculate zoom
    _, _, _, total_size = viewer._get_scene_parameters()
    dim1_diff = dim1_max - dim1_min
    dim2_diff = dim2_max - dim2_min
    # calculate average zoom based on the size of the rectangle
    zoom = np.min(total_size / (dim2_diff, dim1_diff))
    # ensure zoom is a valid number
    if np.isinf(zoom) or np.isnan(zoom) or zoom == 0:
        zoom = 1
    return zoom, dim2_center, dim1_center
