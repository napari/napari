from __future__ import annotations

import napari
from napari.layers.base._base_constants import Mode
from napari.layers.image.image import Image
from napari.layers.utils.interactivity_utils import (
    orient_plane_normal_around_cursor,
)


def orient_plane_normal_along_z(layer: Image):
    orient_plane_normal_around_cursor(layer, plane_normal=(1, 0, 0))


def orient_plane_normal_along_y(layer: Image):
    orient_plane_normal_around_cursor(layer, plane_normal=(0, 1, 0))


def orient_plane_normal_along_x(layer: Image):
    orient_plane_normal_around_cursor(layer, plane_normal=(0, 0, 1))


def hold_to_orient_plane_normal_along_view_direction(layer: Image):
    viewer = napari.viewer.current_viewer()
    if viewer is None or viewer.dims.ndisplay != 3:
        return

    def sync_plane_normal_with_view_direction(event=None):
        """Plane normal syncronisation mouse callback."""
        layer.plane.normal = layer._world_to_displayed_data_ray(
            viewer.camera.view_direction, [-3, -2, -1]
        )

    # update plane normal and add callback to mouse drag
    sync_plane_normal_with_view_direction()
    viewer.camera.events.angles.connect(sync_plane_normal_with_view_direction)
    yield
    # remove callback on key release
    viewer.camera.events.angles.disconnect(
        sync_plane_normal_with_view_direction
    )


def orient_plane_normal_along_view_direction(layer: Image):
    viewer = napari.viewer.current_viewer()
    if viewer is None or viewer.dims.ndisplay != 3:
        return
    layer.plane.normal = layer._world_to_displayed_data_ray(
        viewer.camera.view_direction, [-3, -2, -1]
    )


def activate_image_transform_mode(layer: Image):
    layer.mode = str(Mode.TRANSFORM)


def activate_image_pan_zoom_mode(layer: Image):
    layer.mode = str(Mode.PAN_ZOOM)


image_fun_to_mode = [
    (activate_image_pan_zoom_mode, Mode.PAN_ZOOM),
    (activate_image_transform_mode, Mode.TRANSFORM),
]
