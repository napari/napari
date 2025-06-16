"""Vispy zoom box overlay."""

from __future__ import annotations

import numpy as np

from napari.viewer import ViewerModel


def _get_dim_info(
    mins: np.ndarray, maxs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate center and difference for single dimension."""
    center = (mins + maxs) / 2
    spread = maxs - mins
    spread[spread == 0] = 1.0  # avoid division by zero
    return center, spread


def _get_data_extents(
    data_positions: tuple[tuple[float, ...], tuple[float, ...]],
    displayed: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Get the extents of the overlay in the scene coordinates.

    Parameters
    ----------
    displayed : tuple[int, ...]
        Axes that are currently displayed in the viewer.

    Returns
    -------
    mins : np.ndarray
        Minimum values of the extents in the scene coordinates.
    maxs : np.ndarray
        Maximum values of the extents in the scene coordinates.
    """
    top_left_, bot_right_ = data_positions
    top_left = np.array([top_left_[i] for i in displayed])
    bot_right = np.array([bot_right_[i] for i in displayed])
    extents = np.vstack((top_left, bot_right))
    mins = np.min(extents, axis=0)
    maxs = np.max(extents, axis=0)
    return mins, maxs


def calculate_zoom_proportion(
    data_positions: tuple[tuple[float, ...], tuple[float, ...]],
    viewer: ViewerModel,
) -> tuple[float, float, float, float]:
    """Calculate zoom for specified region."""
    _, _, _, total_size = viewer._get_scene_parameters()

    # calculate the center of the rectangle
    mins, maxs = _get_data_extents(data_positions, viewer.dims.displayed)
    center, spread = _get_dim_info(mins, maxs)

    # calculate average zoom based on the size of the rectangle
    zoom = np.min(total_size / spread)
    scale_factor = viewer._get_scale_factor(0.05)
    if viewer.dims.ndisplay == 2:
        native_zoom = viewer._get_2d_camera_zoom(total_size, scale_factor)

    else:
        native_zoom = viewer._get_3d_camera_zoom(
            np.vstack((mins, maxs)), spread, scale_factor
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
