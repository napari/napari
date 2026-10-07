import numpy as np

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

    coordinates = mouse_event_to_labels_coordinate(layer, event)
    if layer._mode == Mode.ERASE:
        new_label = layer.colormap.background_value
    else:
        new_label = layer.selected_label

    # right click means we are entering paint-and-fill mode, will be continued by the move
    # callback
    if event.button == 2 and len(event.dims_displayed) == 2:
        brush_stroke = layer._overlays['brush_stroke']
        brush_stroke.position = coordinates
        return

    # on press
    with layer.block_history():
        brush_size_data = layer._get_brush_size_data(event.camera_zoom)
        layer._draw(new_label, coordinates, coordinates, brush_size_data)
        yield

        last_cursor_coord = coordinates
        # on move
        while event.type == 'mouse_move':
            coordinates = mouse_event_to_labels_coordinate(layer, event)
            if coordinates is not None or last_cursor_coord is not None:
                # zoom might have changed in between
                brush_size_data = layer._get_brush_size_data(event.camera_zoom)
                layer._draw(
                    new_label,
                    last_cursor_coord,
                    coordinates,
                    brush_size_data,
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


resize_modifiers = tuple(BRUSH_SIZE_ON_MOUSE_MOVE_MODIFIERS_PARTS)


def _on_modifiers_change():
    global resize_modifiers
    modifiers_setting = (
        get_settings().application.brush_size_on_mouse_move_modifiers
    )
    resize_modifiers = tuple(modifiers_setting.value.split('+'))


def resize_or_continue_stroke(layer, event):
    if all(modifier in event.modifiers for modifier in resize_modifiers):
        yield from resize_on_mouse_move(layer, event)
        return

    brush_stroke = layer._overlays['brush_stroke']
    if brush_stroke.position is not None and len(event.dims_displayed) == 2:
        yield from continue_stroke(layer, event)


def resize_on_mouse_move(layer, event):
    start_pos = np.array(event.pos)
    start_pos_world = np.array(event.position)[event.dims_displayed]
    brush_overlay = layer._overlays['brush_circle']
    brush_overlay._is_resizing = True
    start_brush_size = layer.brush_size
    yield

    while event.type == 'mouse_move' and all(
        modifier in event.modifiers for modifier in resize_modifiers
    ):
        if layer.brush_size_is_canvas_pixels:
            radius_delta = event.pos[0] - start_pos[0]
        else:
            radius_delta = (
                layer.world_to_data(event.position)[-1]
                - layer.world_to_data(start_pos_world)[-1]
            )
        layer.brush_size = start_brush_size + radius_delta * 2
        yield

    brush_overlay._is_resizing = False


def _within_start_radius(start, current, brush_size_data):
    return np.linalg.norm(current - start) <= brush_size_data


def continue_stroke(layer, event):
    has_left_start = False
    stroke_points = []
    radius_factor = get_settings().advanced.paint_fill_completion_radius
    brush_stroke = layer._overlays['brush_stroke']

    if layer._mode == Mode.ERASE:
        new_label = layer.colormap.background_value
    else:
        new_label = layer.selected_label

    last_cursor_coord = mouse_event_to_labels_coordinate(layer, event)
    with layer.block_history():
        # on move
        while brush_stroke.position is not None:
            coordinates = mouse_event_to_labels_coordinate(layer, event)
            if coordinates is not None or last_cursor_coord is not None:
                # zoom might have changed in between
                brush_size_data = layer._get_brush_size_data(event.camera_zoom)
                layer._draw(
                    new_label,
                    last_cursor_coord,
                    coordinates,
                    brush_size_data,
                )
                stroke_points.append(coordinates)
                within_range = _within_start_radius(
                    brush_stroke.position,
                    coordinates,
                    brush_size_data * radius_factor,
                )

                if not within_range:
                    has_left_start = True
                elif has_left_start:
                    # done painting, return to start and fill the center
                    layer._draw(
                        new_label,
                        coordinates,
                        brush_stroke.position,
                        brush_size_data,
                    )
                    layer.paint_polygon(stroke_points, layer.selected_label)
                    brush_stroke.position = None
            last_cursor_coord = coordinates
            yield

    brush_stroke.position = None
