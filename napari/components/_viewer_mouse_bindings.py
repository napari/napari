def dims_scroll(viewer, event):
    """Scroll the dimensions slider."""
    if 'Control' not in event.modifiers:
        return
    if event.native.inverted():
        viewer.dims._scroll_progress += event.delta[1]
    else:
        viewer.dims._scroll_progress -= event.delta[1]
    if abs(viewer.dims._scroll_progress) >= 1:
        for i in range(int(abs(viewer.dims._scroll_progress))):
            if viewer.dims._scroll_progress < 0:
                viewer.dims._increment_dims_left()
            else:
                viewer.dims._increment_dims_right()
        viewer.dims._scroll_progress = 0
