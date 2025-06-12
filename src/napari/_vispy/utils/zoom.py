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
    viewer: ViewerModel,
) -> tuple[float, float, float, float]:
    """Calculate zoom for specified region."""
    extent, _, _, total_size = viewer._get_scene_parameters()

    # calculate the center of the rectangle
    z_center = viewer.camera.center[0]
    y_min, y_max, x_min, x_max = viewer._zoom_box.bound_extents()
    y_center, y_spread = _get_dim_info(y_min, y_max)
    x_center, x_spread = _get_dim_info(x_min, x_max)
    # using the viewer's scene size to calculate zoom
    y_prop, x_prop = np.array((y_spread, x_spread)) / viewer._canvas_size
    y_center_prop, x_center_prop = (
        np.array((y_center, x_center)) / viewer._canvas_size
    )
    # print('dim1', y_min, y_max, y_center, y_spread, y_prop, y_center_prop)
    # print('dim2', x_min, x_max, x_center, x_spread, x_prop, x_center_prop)

    # selected the last two dimensions for zooming
    zoom = np.min(np.array(total_size[-2:]) * (y_prop, x_prop))
    y_center, x_center = np.array(total_size[-2:]) * np.array(
        y_center_prop, x_center_prop
    )
    # calculate average zoom based on the size of the rectangle
    if viewer.dims.ndisplay == 3:
        native_zoom = viewer._get_2d_camera_zoom(total_size, 1.0)
    else:
        native_zoom = viewer._get_3d_camera_zoom(extent, total_size, 1.0)
    # adjust zoom by the native zoom factor
    zoom = zoom * native_zoom
    # ensure zoom is a valid number
    if np.isinf(zoom) or np.isnan(zoom) or zoom == 0:
        zoom = 1
    # print('zoom', zoom)
    # print('center', z_center, y_center, x_center)
    return zoom, z_center, y_center, x_center


# def calculate_zoom_proportion(
#     dim1_min: float,
#     dim1_max: float,
#     dim2_min: float,
#     dim2_max: float,
#     dim3_min: float,
#     dim3_max: float,
#     viewer: ViewerModel,
# ) -> tuple[float, float, float, float]:
#     """Calculate zoom for specified region."""
#     # calculate the center of the rectangle
#     dim1_center, dim1_spread = _get_dim_info(dim1_min, dim1_max)
#     dim2_center, dim2_spread = _get_dim_info(dim2_min, dim2_max)
#     dim3_center, dim3_spread = _get_dim_info(dim3_min, dim3_max)
#     print('dim1', dim1_min, dim1_max, dim1_center, dim1_spread)
#     print('dim2', dim2_min, dim2_max, dim2_center, dim2_spread)
#     print('dim3', dim3_min, dim3_max, dim3_center, dim3_spread)
#
#     # using the viewer's scene size to calculate zoom
#     dim1_prop, dim2_prop = (
#         np.array((dim1_spread, dim2_spread)) / viewer._canvas_size
#     )
#
#     extent, _, _, total_size = viewer._get_scene_parameters()
#     print('?', extent, total_size)
#
#     # calculate average zoom based on the size of the rectangle
#     if viewer.dims.ndisplay == 3:
#         zoom = np.min(total_size / (1.0, dim2_prop, dim1_prop))
#         native_zoom = viewer._get_2d_camera_zoom(total_size, 1.0)
#     else:
#         zoom = np.min(total_size / (dim2_spread, dim1_spread))
#         native_zoom = viewer._get_3d_camera_zoom(extent, total_size, 1.0)
#     # adjust zoom by the native zoom factor
#     zoom = zoom * native_zoom
#
#     # ensure zoom is a valid number
#     if np.isinf(zoom) or np.isnan(zoom) or zoom == 0:
#         zoom = 1
#     return zoom, dim1_center, dim2_center, dim3_center
