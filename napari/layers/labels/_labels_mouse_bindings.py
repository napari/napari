import numpy as np

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


class DrawPolygon:
    """Handles mouse events to draw the polygon and updates the overlay.

    Mouse events handled in call:
    - Mouse move: Continuously redraw the latest polygon point with the current mouse position.
    - Mouse press (left button): Adds the current mouse position as a new polygon point.
    - Mouse double click (left button): If there are at least three points in the polygon,
                  the polygon is painted on the image using the current label.
    - Mouse press (right button): Removes the most recent polygon point from the list.
    """

    def __init__(self):
        self._points = []

    def __call__(self, layer, event):
        polygon_overlay = layer._overlays['draw_polygon']
        pos = mouse_event_to_labels_coordinate(layer, event)
        if pos is None:
            return

        pos = np.array(pos, dtype=float)
        dims_displayed = list(event.dims_displayed)
        pos[dims_displayed] += 0.5

        if event.button is None:  # on mouse move
            if self._points:
                polygon_overlay.points = self._points + [pos.tolist()]
        elif (
            event.button == 1 and event.type == 'mouse_press'
        ):  # on mouse left click
            # recenter the point in the center of the image pixel
            pos[dims_displayed] = np.floor(pos[dims_displayed]) + 0.5

            self._points.append(pos.tolist())
            polygon_overlay.points = self._points
        elif (
            event.button == 1 and event.type == 'mouse_double_click'
        ):  # on mouse left double click
            layer._complete_polygon_drawing()
        elif event.button == 2 and self._points:  # on mouse right click
            self._points.pop()
            if self._points:
                polygon_overlay.points = self._points + [pos.tolist()]
            else:
                layer._reset_draw_polygon()

        polygon_overlay.visible = len(self._points) > 0

    def reset(self):
        self._points = []


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
