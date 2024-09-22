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
    """Zoom on double click."""
    if viewer.layers.selection.active.mode != 'pan_zoom':
        return
    if 'Alt' in event.modifiers:
        viewer.camera.zoom /= 2
    else:
        viewer.camera.zoom *= 2
