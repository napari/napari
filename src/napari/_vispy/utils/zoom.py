"""Vispy zoom box overlay."""

from __future__ import annotations

import numpy as np

from napari.viewer import ViewerModel


def _calculate_zoom_for_dimension(dim_min: float, dim_max: float) -> float:
    """Calculate center and difference for single dimension."""
    dim_center = (dim_min + dim_max) / 2
    dim_diff = dim_max - dim_min
    return dim_center, dim_diff


def calculate_zoom_proportion(
    dim1_min: float,
    dim1_max: float,
    dim2_min: float,
    dim2_max: float,
    dim3_min: float,
    dim3_max: float,
    viewer: ViewerModel,
) -> tuple[float, float, float, float]:
    """Calculate zoom for specified region."""
    # calculate the center of the rectangle
    dim1_center, dim1_diff = _calculate_zoom_for_dimension(dim1_min, dim1_max)
    dim2_center, dim2_diff = _calculate_zoom_for_dimension(dim2_min, dim2_max)
    dim3_center, dim3_diff = _calculate_zoom_for_dimension(dim3_min, dim3_max)

    # using the viewer's scene size to calculate zoom
    _, _, _, total_size = viewer._get_scene_parameters()

    # calculate average zoom based on the size of the rectangle
    if viewer.dims.ndisplay == 3:
        zoom = np.min(total_size / (dim3_diff, dim2_diff, dim1_diff))
    else:
        zoom = np.min(total_size / (dim2_diff, dim1_diff))

    # ensure zoom is a valid number
    if np.isinf(zoom) or np.isnan(zoom) or zoom == 0:
        zoom = 1
    return zoom, dim1_center, dim2_center, dim3_center
