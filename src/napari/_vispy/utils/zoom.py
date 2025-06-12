"""Vispy zoom box overlay."""

from __future__ import annotations

import numpy as np

from napari.viewer import ViewerModel


def _get_dim_info(dim_min: float, dim_max: float) -> float:
    """Calculate center and difference for single dimension."""
    dim_center = (dim_min + dim_max) / 2
    dim_spread = dim_max - dim_min
    return dim_center, dim_spread


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
    dim1_center, dim1_spread = _get_dim_info(dim1_min, dim1_max)
    dim2_center, dim2_spread = _get_dim_info(dim2_min, dim2_max)
    dim3_center, dim3_spread = _get_dim_info(dim3_min, dim3_max)

    # using the viewer's scene size to calculate zoom
    extent, _, _, total_size = viewer._get_scene_parameters()

    # calculate average zoom based on the size of the rectangle
    if viewer.dims.ndisplay == 3 and len(total_size) == 3:
        dim3_spread = dim3_spread or 1.0
        zoom = np.min(total_size / (dim3_spread, dim2_spread, dim1_spread))
        native_zoom = viewer._get_2d_camera_zoom(total_size, 1.0)
    else:
        zoom = np.min(total_size / (dim2_spread, dim1_spread))
        native_zoom = viewer._get_3d_camera_zoom(extent, total_size, 1.0)
    # adjust zoom by the native zoom factor
    zoom = zoom * native_zoom

    # ensure zoom is a valid number
    if np.isinf(zoom) or np.isnan(zoom) or zoom == 0:
        zoom = 1
    return zoom, dim1_center, dim2_center, dim3_center
