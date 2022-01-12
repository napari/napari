from __future__ import annotations

from ...layers.utils.layer_utils import register_layer_action
from ...utils.translations import trans
from ._image_constants import Mode
from .image import Image


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
