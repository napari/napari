from __future__ import annotations

import napari

from ...layers.utils.interactivity_utils import (
    orient_plane_normal_around_cursor,
)
from ...layers.utils.layer_utils import register_layer_action
from ...utils.translations import trans
from ._image_constants import Mode
from .image import Image


def register_image_action(description: str):
    return register_layer_action(Image, description=description)


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


@Image.bind_key('o')
@register_image_action(
    trans._('orient plane normal along camera view direction')
)
def orient_plane_normal_along_view_direction(layer: Image):
    if napari.current_viewer().dims.ndisplay != 3:
        return
    view_direction = napari.current_viewer().camera.view_direction
    layer.plane.normal = layer._world_to_data_ray(view_direction)


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


def register_image_action(description):
    return register_layer_action(Image, description)


@register_image_action(trans._('Transform'))
def activate_image_select_mode(layer):
    layer.mode = Mode.TRANSFORM


@register_image_action(trans._('Pan/zoom'))
def activate_image_pan_zoom_mode(layer):
    layer.mode = Mode.PAN_ZOOM
