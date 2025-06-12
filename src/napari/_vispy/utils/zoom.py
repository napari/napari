"""Vispy zoom box overlay."""

from __future__ import annotations

import numpy as np

from napari.viewer import ViewerModel


def _get_dim_info(mins: np.ndarray, maxs: np.ndarray) -> float:
    """Calculate center and difference for single dimension."""
    center = (mins + maxs) / 2
    spread = maxs - mins
    return center, spread


def calculate_zoom_proportion(
    viewer: ViewerModel,
) -> tuple[float, float, float, float]:
    """Calculate zoom for specified region."""
    extent, _, _, total_size = viewer._get_scene_parameters()

    # calculate the center of the rectangle
    mins, maxs = viewer._zoom_box.data_extents(viewer.dims.displayed)
    center, spread = _get_dim_info(mins, maxs)

    # calculate average zoom based on the size of the rectangle
    zoom = np.min(total_size / spread)
    scale_factor = viewer._get_scale_factor(0.05)
    if viewer.dims.ndisplay == 3:
        native_zoom = viewer._get_2d_camera_zoom(total_size, scale_factor)

    else:
        native_zoom = viewer._get_3d_camera_zoom(
            extent, total_size, scale_factor
        )

    # handle 3-D display with 2-D data and 2-D display with N-D data
    if len(center) == 3:
        z_center, x_center, y_center = center
    else:
        z_center = 1.0
        x_center, y_center = center[-2:]

    # adjust zoom by the native zoom factor
    zoom = zoom * native_zoom
    # ensure zoom is a valid number
    if np.isinf(zoom) or np.isnan(zoom) or zoom < 1:
        zoom = 1
    return zoom, z_center, y_center, x_center
