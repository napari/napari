from napari.layers.labels._labels_constants import Mode
from napari.layers.labels._labels_utils import mouse_event_to_labels_coordinate
from napari.settings import get_settings


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

    # Do not allow drawing while adjusting the brush size with the mouse
    if layer.cursor == 'circle_frozen':
        return

    coordinates = mouse_event_to_labels_coordinate(layer, event)
    if layer._mode == Mode.ERASE:
        new_label = layer._background_label
    else:
        new_label = layer.selected_label

    # on press
    with layer.block_history():
        layer._draw(new_label, coordinates, coordinates)
        yield

        last_cursor_coord = coordinates
        # on move
        while event.type == 'mouse_move':
            coordinates = mouse_event_to_labels_coordinate(layer, event)
            if coordinates is not None or last_cursor_coord is not None:
                layer._draw(new_label, last_cursor_coord, coordinates)
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


class BrushSizeOnMouseMove:
    """Enables changing the brush size by moving the mouse while holding down the specified modifiers

    When hold down specified modifiers and move the mouse,
    the callback will adjust the brush size based on the direction of the mouse movement.
    Moving the mouse right will increase the brush size, while moving it left will decrease it.
    The amount of change is proportional to the distance moved by the mouse.

    Parameters
    ----------
    min_brush_size : int
        The minimum brush size.

    """

    def __init__(self, min_brush_size: int = 1):
        self.min_brush_size = min_brush_size
        self.init_pos = None
        self.init_brush_size = None

        get_settings().application.events.brush_size_on_mouse_move_modifiers.connect(
            self._on_modifiers_change
        )
        self._on_modifiers_change()

    def __call__(self, layer, event):
        if all(modifier in event.modifiers for modifier in self.modifiers):
            pos = event.pos  # position in the canvas coordinates (x, y)

            if self.init_pos is None:
                self.init_pos = pos
                self.init_brush_size = layer.brush_size
                layer.cursor = 'circle_frozen'
            else:
                brush_size_delta = round(
                    (pos[0] - self.init_pos[0]) / event.camera_zoom
                )
                new_brush_size = self.init_brush_size + brush_size_delta

                bounded_brush_size = max(new_brush_size, self.min_brush_size)
                layer.brush_size = bounded_brush_size
        else:
            self.init_pos = None
            if layer.cursor == 'circle_frozen':
                layer.cursor = 'circle'

    def _on_modifiers_change(self):
        modifiers_setting = (
            get_settings().application.brush_size_on_mouse_move_modifiers
        )
        self.modifiers = modifiers_setting.value.split('+')
