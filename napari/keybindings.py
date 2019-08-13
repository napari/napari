import numpy as np

from .viewer import Viewer


@Viewer.bind_key('Control-F')
def toggle_fullscreen(viewer):
    if viewer.window._qt_window.isFullScreen():
        viewer.window._qt_window.showNormal()
    else:
        viewer.window._qt_window.showFullScreen()


@Viewer.bind_key('Control-T')
def toggle_theme(viewer):
    theme_names = list(viewer.themes.keys())
    cur_theme = theme_names.index(viewer.theme)
    viewer.theme = theme_names[(cur_theme + 1) % len(theme_names)]


@Viewer.bind_key('Left')
def increment_dims_left(viewer):
    axis = viewer.window.qt_viewer.dims.last_used
    if axis is not None:
        cur_point = viewer.dims.point[axis]
        axis_range = viewer.dims.range[axis]
        new_point = np.clip(
            cur_point - axis_range[2],
            axis_range[0],
            axis_range[1] - axis_range[2],
        )
        viewer.dims.set_point(axis, new_point)


@Viewer.bind_key('Right')
def increment_dims_right(viewer):
    axis = viewer.window.qt_viewer.dims.last_used
    if axis is not None:
        cur_point = viewer.dims.point[axis]
        axis_range = viewer.dims.range[axis]
        new_point = np.clip(
            cur_point + axis_range[2],
            axis_range[0],
            axis_range[1] - axis_range[2],
        )
        viewer.dims.set_point(axis, new_point)


@Viewer.bind_key('Up')
def dims_focus_up(viewer):
    displayed = list(np.nonzero(viewer.window.qt_viewer.dims._displayed)[0])
    if len(displayed) == 0:
        return

    axis = viewer.window.qt_viewer.dims.last_used
    if axis is None:
        viewer.window.qt_viewer.dims.last_used = displayed[-1]
    else:
        index = (displayed.index(axis) + 1) % len(displayed)
        viewer.window.qt_viewer.dims.last_used = displayed[index]


@Viewer.bind_key('Down')
def dims_focus_down(viewer):
    displayed = list(np.nonzero(viewer.window.qt_viewer.dims._displayed)[0])
    if len(displayed) == 0:
        return

    axis = viewer.window.qt_viewer.dims.last_used
    if axis is None:
        viewer.window.qt_viewer.dims.last_used = displayed[0]
    else:
        index = (displayed.index(axis) - 1) % len(displayed)
        viewer.window.qt_viewer.dims.last_used = displayed[index]


Viewer.bind_key('Control-Backspace', lambda v: v.layers.remove_selected())
Viewer.bind_key('Control-A', lambda v: v.layers.select_all())
Viewer.bind_key('Control-[', lambda v: v.layers.select_previous())
Viewer.bind_key('Control-]', lambda v: v.layers.select_next())
Viewer.bind_key('Control-R', lambda v: v.reset_view())
