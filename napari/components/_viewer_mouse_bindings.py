def dims_scroll(viewer, cursor_event):
    """Scroll the dimensions slider.

    Parameters
    ----------
    viewer : napari.viewer.Viewer
        The napari viewer.
    cursor_event : napari.components.cursor_event.CursorEvent
        Cursor event.
    """
    if 'Control' not in cursor_event.modifiers:
        return
    if cursor_event.inverted:
        viewer.dims._scroll_progress += cursor_event.delta[1]
    else:
        viewer.dims._scroll_progress -= cursor_event.delta[1]
    while abs(viewer.dims._scroll_progress) >= 1:
        if viewer.dims._scroll_progress < 0:
            viewer.dims._increment_dims_left()
            viewer.dims._scroll_progress += 1
        else:
            viewer.dims._increment_dims_right()
            viewer.dims._scroll_progress -= 1
