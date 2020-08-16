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


@Labels.bind_key('P')
def activate_paint_mode(layer):
    """Activate the paintbrush."""
    layer.mode = Mode.PAINT


@Labels.bind_key('F')
def activate_fill_mode(layer):
    """Activate the fill bucket."""
    layer.mode = Mode.FILL


@Labels.bind_key('Z')
def activate_pan_zoom_mode(layer):
    """Activate pan and zoom mode."""
    layer.mode = Mode.PAN_ZOOM


@Labels.bind_key('L')
def activate_picker_mode(layer):
    """Activate the label picker."""
    layer.mode = Mode.PICK


@Labels.bind_key('E')
def erase(layer):
    """Activate the label eraser."""
    layer.mode = Mode.ERASE


@Labels.bind_key('M')
def new_label(layer):
    """Set the currently selected label to the largest used label plus one."""
    layer.selected_label = layer.data.max() + 1


@Labels.bind_key('D')
def decrease_label_id(layer):
    """Decrease the currently selected label by one."""
    layer.selected_label -= 1


@Labels.bind_key('I')
def increase_label_id(layer):
    """Increase the currently selected label by one."""
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
