import numpy as np

# This is the minimum size of the zoom box (in pixels) that will
# trigger a zoom when the user drags the mouse while holding Alt.
MIN_ZOOMBOX_SIZE = 5


def dims_scroll(viewer, event):
    """Scroll the dimensions slider."""
    if 'Control' not in event.modifiers:
        return
    if event.native.inverted():
        viewer.dims._scroll_progress += event.delta[1]
    else:
        viewer.dims._scroll_progress -= event.delta[1]
    while abs(viewer.dims._scroll_progress) >= 1:
        if viewer.dims._scroll_progress < 0:
            viewer.dims._increment_dims_left()
            viewer.dims._scroll_progress += 1
        else:
            viewer.dims._increment_dims_right()
            viewer.dims._scroll_progress -= 1


def double_click_to_zoom(viewer, event):
    """Zoom in on double click by zoom_factor; zoom out with Alt."""
    if (
        viewer.layers.selection.active
        and viewer.layers.selection.active.mode != 'pan_zoom'
    ):
        return
    # if Alt held down, zoom out instead
    zoom_factor = 0.5 if 'Alt' in event.modifiers else 2
    viewer.camera.zoom *= zoom_factor
    if viewer.dims.ndisplay == 3 and viewer.dims.ndim == 3:
        viewer.camera.center = np.asarray(viewer.camera.center) + (
            np.asarray(event.position)[np.asarray(viewer.dims.displayed)]
            - np.asarray(viewer.camera.center)
        ) * (1 - 1 / zoom_factor)
    else:
        viewer.camera.center = np.asarray(viewer.camera.center)[-2:] + (
            np.asarray(event.position)[-2:]
            - np.asarray(viewer.camera.center)[-2:]
        ) * (1 - 1 / zoom_factor)


def drag_to_zoom(viewer, event):
    """While holding Alt, drag mouse to select a region to zoom.

    This function allows the user to click and drag the mouse while
    holding the `Alt` key to create a zoom box. When the mouse is released,
    the camera zooms into the selected region.
    """
    if 'Alt' not in event.modifiers or event.type != 'mouse_press':
        return

    # on mouse press
    viewer._zoom_box.visible = True
    press_pos = event.pos[::-1]
    press_position = event.position
    move_pos = press_pos
    move_position = press_position
    viewer._zoom_box.position = (press_pos, press_pos)
    yield
    event.handled = True

    # on mouse move
    while event.type == 'mouse_move':
        if 'Alt' not in event.modifiers:
            viewer._zoom_box.visible = False
            yield
            return

        move_pos = event.pos[::-1]
        viewer._zoom_box.position = (press_pos, move_pos)
        move_position = event.position
        yield

    # on mouse release
    viewer._zoom_box.visible = False

    # only trigger zoom if the box is larger than a MIN_ZOOMBOX_SIZE in pixels
    distance = np.abs(np.array(press_pos) - np.array(move_pos))
    if distance.min() > MIN_ZOOMBOX_SIZE:
        # Slice to the last two coordinates (displayed axes) for cases where
        # ndim>2 and ndisplay=2
        viewer._zoom_box.zoom_area = (
            press_position[-2:],
            move_position[-2:],
        )
    yield
