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

    # define a mouse drag callback to sync plane normal during mouse drag
    camera = napari.current_viewer().camera

    def sync_plane_normal_with_view_direction(layer, event=None):
        yield
        while event.type == 'mouse_move':
            view_direction = camera.view_direction
            layer.plane.normal = layer._world_to_data_ray(view_direction)
            yield

    # update plane normal and add callback to mouse drag
    layer.plane.normal = layer._world_to_data_ray(camera.view_direction)
    layer.mouse_drag_callbacks.append(sync_plane_normal_with_view_direction)
    yield
    # remove callback on key release
    layer.mouse_drag_callbacks.remove(sync_plane_normal_with_view_direction)


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
