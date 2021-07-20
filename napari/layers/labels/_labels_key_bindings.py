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
    layer.selected_label = layer.data.max() + 1


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


@Labels.bind_key('Shift')
def preserve_labels(layer):
    """Toggle preserve label option when pressed."""
    # on key press
    layer.preserve_labels = not layer.preserve_labels

    yield

    # on key release
    layer.preserve_labels = not layer.preserve_labels


@Labels.bind_key('Control')
def switch_fill(layer):
    """Switch to fill mode temporarily when pressed."""
    previous_mode = layer.mode

    # on key press
    layer.mode = Mode.FILL

    yield

    # on key release
    layer.mode = previous_mode


@Labels.bind_key('Alt')
def switch_erase(layer):
    """Switch to erase mode temporarily when pressed."""
    previous_mode = layer.mode

    # on key press
    layer.mode = Mode.ERASE

    yield

    # on key release
    layer.mode = previous_mode


@Labels.bind_key('1')
def change_to_erase(layer):
    """Switch to erase mode"""
    layer.mode = Mode.ERASE


@Labels.bind_key('2')
def change_to_fill(layer):
    """Switch to fill mode"""
    layer.mode = Mode.FILL


@Labels.bind_key('3')
def change_to_paint(layer):
    """Switch to paint mode"""
    layer.mode = Mode.PAINT


@Labels.bind_key('4')
def change_to_pick(layer):
    """Switch to pick mode"""
    layer.mode = Mode.PICK


@Labels.bind_key('5')
def change_to_pan_zoom(layer):
    """Switch to pan zoom mode"""
    layer.mode = Mode.PAN_ZOOM


@Labels.bind_key('-')
def decrement_label(layer):
    """Decrement the value of the selected label if said label is > 0"""
    if layer.selected_label > 0:
        layer.selected_label -= 1


@Labels.bind_key('=')
def increment_label(layer):
    """Increment the value of the selected label"""
    layer.selected_label += 1
