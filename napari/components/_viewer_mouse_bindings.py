def dims_scroll(viewer, cursor):
    """Scroll the dimensions slider."""
    if 'Control' not in cursor.modifiers:
        return
    if cursor.inverted:
        viewer.dims._scroll_progress += cursor.delta[0]
    else:
        viewer.dims._scroll_progress -= cursor.delta[0]
    while abs(viewer.dims._scroll_progress) >= 1:
        if viewer.dims._scroll_progress < 0:
            viewer.dims._increment_dims_left()
            viewer.dims._scroll_progress += 1
        else:
            viewer.dims._increment_dims_right()
            viewer.dims._scroll_progress -= 1
