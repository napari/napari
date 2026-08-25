from napari.layers.labels._labels_constants import Mode
from napari.layers.labels._labels_utils import mouse_event_to_labels_coordinate
from napari.settings import get_settings

BRUSH_SIZE_ON_MOUSE_MOVE_MODIFIERS_PARTS = ('Alt',)


def change_brush_size_on_mouse_move_modifiers(value: tuple[str]) -> None:
    """Update the brush size on mouse move modifiers from settings."""
    global BRUSH_SIZE_ON_MOUSE_MOVE_MODIFIERS_PARTS

    BRUSH_SIZE_ON_MOUSE_MOVE_MODIFIERS_PARTS = value


def draw(layer, event):
    """Draw with the currently selected label to a coordinate.

    This method have different behavior when draw is called
    with different labeling layer mode.

    In PAINT mode the cursor functions like a paint brush changing any
    pixels it brushes over to the current label. If the background label
    `0` is selected than any pixels will be changed to background and this
    tool functions like an eraser. The size and shape of the cursor can be
    adjusted in the properties widget.

    In FILL mode the cursor functions like a fill bucket replacing pixels
    of the label clicked on with the current label. It can either replace
    all pixels of that label or just those that are contiguous with the
    clicked on pixel. If the background label `0` is selected than any
    pixels will be changed to background and this tool functions like an
    eraser
    """

    # In PAINT mode the right button (and any click during an active stroke) is
    # reserved for the encircle-and-fill brush stroke handled by the
    # brush_stroke overlay.
    brush_stroke = layer._overlays['brush_stroke']
    if brush_stroke.active or (brush_stroke.enabled and event.button == 2):
        return

    coordinates = mouse_event_to_labels_coordinate(layer, event)
    if layer._mode == Mode.ERASE:
        new_label = layer.colormap.background_value
    else:
        new_label = layer.selected_label

    # on press
    with layer.block_history():
        layer._draw(new_label, coordinates, coordinates, event.camera_zoom)
        yield

        last_cursor_coord = coordinates
        # on move
        while event.type == 'mouse_move':
            coordinates = mouse_event_to_labels_coordinate(layer, event)
            if coordinates is not None or last_cursor_coord is not None:
                layer._draw(
                    new_label,
                    last_cursor_coord,
                    coordinates,
                    event.camera_zoom,
                )
            last_cursor_coord = coordinates
            yield


def pick(layer, event):
    """Change the selected label to the same as the region clicked."""
    # on press
    layer.selected_label = (
        layer.get_value(
            event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True,
        )
        or 0
    )


modifiers = tuple(BRUSH_SIZE_ON_MOUSE_MOVE_MODIFIERS_PARTS)


def _on_modifiers_change():
    global modifiers
    modifiers_setting = (
        get_settings().application.brush_size_on_mouse_move_modifiers
    )
    modifiers = tuple(modifiers_setting.value.split('+'))


def resize_brush_on_mouse_move(layer, event):
    if not all(modifier in event.modifiers for modifier in modifiers):
        return

    min_brush_size = 1
    start_pos = event.pos
    start_brush_size = layer.brush_size

    layer._is_resizing_brush = True
    yield

    while event.type == 'mouse_move':
        brush_size_delta = round(
            (event.pos[0] - start_pos[0]) / event.camera_zoom
        )
        new_brush_size = start_brush_size + brush_size_delta

        bounded_brush_size = max(new_brush_size, min_brush_size)
        layer.brush_size = bounded_brush_size

    layer._is_resizing_brush = False
