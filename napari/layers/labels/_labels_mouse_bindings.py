from typing import Optional

from napari.layers.labels._labels_constants import Mode
from napari.layers.labels._labels_utils import mouse_event_to_labels_coordinate


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
    """Enables changing the brush size by moving the mouse while holding down the 'Ctrl+Alt'

    When hold down both the 'Ctrl' and 'Alt' keys and move the mouse,
    the callback will adjust the brush size based on the direction of the mouse movement.
    Moving the mouse right will increase the brush size, while moving it left will decrease it.
    The amount of change is proportional to the distance moved by the mouse.

    Parameters
    ----------
    min_brush_size : int
        The minimum brush size.
    max_brush_size : int
        The maximum brush size.
    sensitivity : float
        Controls the sensitivity of the brush's size change to mouse movement.

    """

    def __init__(
        self,
        min_brush_size: int = 1,
        max_brush_size: Optional[int] = None,
        sensitivity: float = 0.2,
    ):
        self.min_brush_size = min_brush_size
        self.max_brush_size = max_brush_size
        self.sensitivity = sensitivity
        self.init_pos = None
        self.init_brush_size = None
        self.accumulated_delta = 0

    def __call__(self, layer, event):
        if 'Control' in event.modifiers and 'Alt' in event.modifiers:
            # The Qt cursor is used to keep the cursor position frozen while the callback is active
            qt_cursor = event.source.native.cursor()
            pos = qt_cursor.pos()

            if self.init_pos is None:
                self.init_pos = pos
                self.init_brush_size = layer.brush_size
                self.accumulated_delta = 0
            else:
                # To minimize rounding errors after multiplying by the sensitivity,
                # it is important to aggregate the delta change in a distinct variable
                self.accumulated_delta += pos.x() - self.init_pos.x()
                qt_cursor.setPos(self.init_pos)

                brush_size_delta = round(
                    self.sensitivity * self.accumulated_delta
                )
                new_brush_size = self.init_brush_size + brush_size_delta

                bounded_brush_size = max(new_brush_size, self.min_brush_size)
                if self.max_brush_size is not None:
                    bounded_brush_size = min(
                        bounded_brush_size, self.max_brush_size
                    )

                layer.brush_size = bounded_brush_size

                # Reset the delta when the brush size goes outside its limits
                if new_brush_size != bounded_brush_size:
                    self.init_brush_size = layer.brush_size
                    self.accumulated_delta = 0
        else:
            self.init_pos = None
