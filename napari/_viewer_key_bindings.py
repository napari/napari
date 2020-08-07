from .viewer import Viewer


@Viewer.bind_key('Control-F')
def toggle_fullscreen(viewer):
    """Toggle fullscreen mode."""
    if viewer.window._qt_window.isFullScreen():
        viewer.window._qt_window.showNormal()
    else:
        viewer.window._qt_window.showFullScreen()


@Viewer.bind_key('Alt-Up')
def focus_axes_up(viewer):
    """Move focus of dimensions slider up."""
    viewer.window.qt_viewer.dims.focus_up()


@Viewer.bind_key('Alt-Down')
def focus_axes_down(viewer):
    """Move focus of dimensions slider down."""
    viewer.window.qt_viewer.dims.focus_down()


@Viewer.bind_key('Control-Alt-P')
def play(viewer):
    """Toggle animation on the first axis"""
    if viewer.window.qt_viewer.dims.is_playing:
        viewer.window.qt_viewer.dims.stop()
    else:
        axis = viewer.window.qt_viewer.dims.last_used or 0
        viewer.window.qt_viewer.dims.play(axis)
