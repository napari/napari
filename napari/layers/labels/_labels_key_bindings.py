import numpy as np

from napari.layers.labels._labels_constants import Mode
from napari.layers.labels.labels import Labels

MIN_BRUSH_SIZE = 1
MAX_BRUSH_SIZE = 40


def hold_to_pan_zoom(layer: Labels):
    """Hold to pan and zoom in the viewer."""
    if layer._mode != Mode.PAN_ZOOM:
        # on key press
        prev_mode = layer.mode
        layer.mode = Mode.PAN_ZOOM

        yield

        # on key release
        layer.mode = prev_mode


def activate_paint_mode(layer: Labels):
    layer.mode = Mode.PAINT


def activate_fill_mode(layer: Labels):
    layer.mode = Mode.FILL


def activate_label_pan_zoom_mode(layer: Labels):
    layer.mode = Mode.PAN_ZOOM


def activate_label_picker_mode(layer: Labels):
    """Activate the label picker."""
    layer.mode = Mode.PICK


def activate_label_erase_mode(layer: Labels):
    layer.mode = Mode.ERASE


labels_fun_to_mode = [
    (activate_label_erase_mode, Mode.ERASE),
    (activate_paint_mode, Mode.PAINT),
    (activate_fill_mode, Mode.FILL),
    (activate_label_picker_mode, Mode.PICK),
    (activate_label_pan_zoom_mode, Mode.PAN_ZOOM),
]


def new_label(layer: Labels):
    """Set the currently selected label to the largest used label plus one."""
    layer.selected_label = np.max(layer.data) + 1


def decrease_label_id(layer: Labels):
    layer.selected_label -= 1


def increase_label_id(layer: Labels):
    layer.selected_label += 1


def decrease_brush_size(layer: Labels):
    """Decrease the brush size"""
    if (
        layer.brush_size > MIN_BRUSH_SIZE
    ):  # here we should probably add a non-hard-coded
        # reference to the limit values of brush size?
        layer.brush_size -= 1


def increase_brush_size(layer: Labels):
    """Increase the brush size"""
    if (
        layer.brush_size < MAX_BRUSH_SIZE
    ):  # here we should probably add a non-hard-coded
        # reference to the limit values of brush size?
        layer.brush_size += 1


def toggle_preserve_labels(layer: Labels):
    layer.preserve_labels = not layer.preserve_labels


def _get_preserve_labels_toggled(layer: Labels):
    """Whether 'preserve labels' should appear toggled (e.g. in menu items)"""
    return layer.preserve_labels


def undo(layer: Labels):
    """Undo the last paint or fill action since the view slice has changed."""
    layer.undo()


def redo(layer: Labels):
    """Redo any previously undone actions."""
    layer.redo()
