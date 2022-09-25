from __future__ import annotations

import napari

from ...layers.utils.interactivity_utils import (
    orient_plane_normal_around_cursor,
)
from ...layers.utils.layer_utils import register_layer_action
from ...utils.translations import trans
from ._image_constants import Mode
from .image import Image


def register_image_action(description: str, repeatable: bool = False):
    return register_layer_action(Image, description, repeatable)


@Image.bind_key('z')
@register_image_action(trans._('Orient plane normal along z-axis'))
def orient_plane_normal_along_z(layer: Image):
    orient_plane_normal_around_cursor(layer, plane_normal=(1, 0, 0))


@Image.bind_key('y')
@register_image_action(trans._('orient plane normal along y-axis'))
def orient_plane_normal_along_y(layer: Image):
    orient_plane_normal_around_cursor(layer, plane_normal=(0, 1, 0))


@Image.bind_key('x')
@register_image_action(trans._('orient plane normal along x-axis'))
def orient_plane_normal_along_x(layer: Image):
    orient_plane_normal_around_cursor(layer, plane_normal=(0, 0, 1))


@register_image_action(trans._('orient plane normal along view direction'))
def orient_plane_normal_along_view_direction(layer: Image):
    viewer = napari.viewer.current_viewer()
    if viewer.dims.ndisplay != 3:
        return
    layer.plane.normal = layer._world_to_displayed_data_ray(
        viewer.camera.view_direction, dims_displayed=[-3, -2, -1]
    )


@Image.bind_key('o')
def synchronise_plane_normal_with_view_direction(layer: Image):
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


@Image.bind_key('Space')
def hold_to_pan_zoom(layer):
    """Hold to pan and zoom in the viewer."""
    if layer._mode != Mode.PAN_ZOOM:
        # on key press
        prev_mode = layer.mode
        layer.mode = Mode.PAN_ZOOM

        yield

        # on key release
        layer.mode = prev_mode


@register_image_action(trans._('Transform'))
def activate_image_select_mode(layer):
    layer.mode = Mode.TRANSFORM


@register_image_action(trans._('Pan/zoom'))
def activate_image_pan_zoom_mode(layer):
    layer.mode = Mode.PAN_ZOOM
