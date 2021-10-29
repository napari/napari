import numpy as np

from ...layers.utils.layer_utils import register_layer_action
from ...utils.translations import trans
from ._labels_constants import Mode
from .labels import Labels


@Labels.bind_key('Space')
def hold_to_pan_zoom(layer):
    """Hold to pan and zoom in the viewer."""
    if layer._mode != Mode.PAN_ZOOM:
        # on key press
        prev_mode = layer.mode
        layer.mode = Mode.PAN_ZOOM

        yield

        # on key release
        layer.mode = prev_mode


def register_label_action(description):
    return register_layer_action(Labels, description)


@register_label_action(trans._("Activate the paint brush"))
def activate_paint_mode(layer):
    layer.mode = Mode.PAINT


@register_label_action(trans._("Activate the fill bucket"))
def activate_fill_mode(layer):
    layer.mode = Mode.FILL


@register_label_action(trans._('Pan/zoom mode'))
def activate_label_pan_zoom_mode(layer):
    layer.mode = Mode.PAN_ZOOM


@register_label_action(trans._('Pick mode'))
def activate_label_picker_mode(layer):
    """Activate the label picker."""
    layer.mode = Mode.PICK


@register_label_action(trans._("Activate the label eraser"))
def activate_label_erase_mode(layer):
    layer.mode = Mode.ERASE


@register_label_action(
    trans._(
        "Set the currently selected label to the largest used label plus one."
    ),
)
def new_label(layer):
    """Set the currently selected label to the largest used label plus one."""
    layer.selected_label = np.max(layer.data) + 1


@register_label_action(
    trans._("Decrease the currently selected label by one."),
)
def decrease_label_id(layer):
    layer.selected_label -= 1


@register_label_action(
    trans._("Increase the currently selected label by one."),
)
def increase_label_id(layer):
    layer.selected_label += 1


@Labels.bind_key('Control-Z')
def undo(layer):
    """Undo the last paint or fill action since the view slice has changed."""
    layer.undo()


@Labels.bind_key('Control-Shift-Z')
def redo(layer):
    """Redo any previously undone actions."""
    layer.redo()
