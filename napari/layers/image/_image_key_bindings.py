from __future__ import annotations

import napari
from napari.layers.image._image_constants import Mode
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


def orient_plane_normal_along_view_direction(layer: Image):
    viewer = napari.viewer.current_viewer()
    if viewer.dims.ndisplay != 3:
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


def hold_to_pan_zoom(layer: Image):
    """Hold to pan and zoom in the viewer."""
    if layer._mode != Mode.PAN_ZOOM:
        # on key press
        prev_mode = layer.mode
        layer.mode = Mode.PAN_ZOOM

        yield

        # on key release
        layer.mode = prev_mode


def activate_image_transform_mode(layer: Image):
    layer.mode = Mode.TRANSFORM


def activate_image_pan_zoom_mode(layer: Image):
    layer.mode = Mode.PAN_ZOOM
