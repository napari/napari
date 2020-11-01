from .viewer import Viewer


@Viewer.bind_key('Alt-Up')
def focus_axes_up(viewer):
    """Move focus of dimensions slider up."""
    viewer.window.qt_viewer.dims.focus_up()


@Viewer.bind_key('Alt-Down')
def focus_axes_down(viewer):
    """Move focus of dimensions slider down."""
    viewer.window.qt_viewer.dims.focus_down()
