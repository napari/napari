from app_model.types import KeyCode

from napari.layers.base._base_constants import Mode
from napari.layers.base.base import Layer


@Layer.bind_key(KeyCode.Space)
def hold_to_pan_zoom(layer):
    """Hold to pan and zoom in the viewer."""
    if layer._mode != Mode.PAN_ZOOM:
        # on key press
        prev_mode = layer.mode
        layer.mode = Mode.PAN_ZOOM

        yield

        # on key release
        layer.mode = prev_mode
