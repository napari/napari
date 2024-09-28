import numpy as np


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
    if viewer.dims.ndisplay == 3:
        viewer.camera.center = np.asarray(viewer.camera.center) + (
            np.asarray(event.position)[np.asarray(viewer.dims.displayed)]
            - np.asarray(viewer.camera.center)
        ) * (1 - 1 / zoom_factor)
    else:
        viewer.camera.center = np.asarray(viewer.camera.center)[-2:] + (
            np.asarray(event.position)[-2:]
            - np.asarray(viewer.camera.center)[-2:]
        ) * (1 - 1 / zoom_factor)
