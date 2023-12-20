import numpy as np

from napari.layers.labels._labels_constants import Mode
from napari.layers.labels.labels import Labels

MIN_BRUSH_SIZE = 1


def activate_labels_transform_mode(layer: Labels):
    layer.mode = Mode.TRANSFORM


def activate_labels_pan_zoom_mode(layer: Labels):
    layer.mode = Mode.PAN_ZOOM


def activate_labels_paint_mode(layer: Labels):
    layer.mode = Mode.PAINT


def activate_labels_polygon_mode(layer: Labels):
    layer.mode = Mode.POLYGON


def activate_labels_fill_mode(layer: Labels):
    layer.mode = Mode.FILL


def activate_labels_picker_mode(layer: Labels):
    """Activate the label picker."""
    layer.mode = Mode.PICK


def activate_labels_erase_mode(layer: Labels):
    layer.mode = Mode.ERASE


labels_fun_to_mode = [
    (activate_labels_pan_zoom_mode, Mode.PAN_ZOOM),
    (activate_labels_transform_mode, Mode.TRANSFORM),
    (activate_labels_erase_mode, Mode.ERASE),
    (activate_labels_paint_mode, Mode.PAINT),
    (activate_labels_polygon_mode, Mode.POLYGON),
    (activate_labels_fill_mode, Mode.FILL),
    (activate_labels_picker_mode, Mode.PICK),
]


def new_label(layer: Labels):
    """Set the currently selected label to the largest used label plus one."""
    layer.selected_label = np.max(layer.data) + 1


def swap_selected_and_background_labels(layer: Labels):
    """Swap between the selected label and the background label."""
    layer.swap_selected_and_background_labels()


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


def reset_polygon(layer: Labels):
    """Reset the drawing of the current polygon."""
    layer._overlays["polygon"].points = []


def complete_polygon(layer: Labels):
    """Complete the drawing of the current polygon."""
    # Because layer._overlays has type Overlay, mypy doesn't know that
    # ._overlays["polygon"] has type LabelsPolygonOverlay, so type ignore for now
    # TODO: Improve typing of layer._overlays to fix this
    layer._overlays["polygon"].add_polygon_to_labels(layer)
