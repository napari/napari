import numpy as np
from copy import copy
from .viewer import Viewer


@Viewer.bind_key('Control-F')
def toggle_fullscreen(viewer):
    if viewer.window._qt_window.isFullScreen():
        viewer.window._qt_window.showNormal()
    else:
        viewer.window._qt_window.showFullScreen()


@Viewer.bind_key('Control-Shift-T')
def toggle_theme(viewer):
    theme_names = list(viewer.themes.keys())
    cur_theme = theme_names.index(viewer.theme)
    viewer.theme = theme_names[(cur_theme + 1) % len(theme_names)]


@Viewer.bind_key('Control-Y')
def toggle_ndisplay(viewer):
    if viewer.dims.ndisplay == 3:
        viewer.dims.ndisplay = 2
    else:
        viewer.dims.ndisplay = 3


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


Viewer.bind_key('Control-E', lambda v: v.dims._roll())
Viewer.bind_key('Control-T', lambda v: v.dims._transpose())
Viewer.bind_key('Alt-Up', lambda v: v.window.qt_viewer.dims.focus_up())
Viewer.bind_key('Alt-Down', lambda v: v.window.qt_viewer.dims.focus_down())
Viewer.bind_key('Control-Backspace', lambda v: v.layers.remove_selected())
Viewer.bind_key('Control-A', lambda v: v.layers.select_all())
Viewer.bind_key(
    'Control-Shift-Backspace',
    lambda v: (v.layers.select_all(), v.layers.remove_selected()),
)
Viewer.bind_key('Up', lambda v: v.layers.select_next())
Viewer.bind_key('Down', lambda v: v.layers.select_previous())
Viewer.bind_key('Shift-Up', lambda v: v.layers.select_next(shift=True))
Viewer.bind_key('Shift-Down', lambda v: v.layers.select_previous(shift=True))
Viewer.bind_key('Control-R', lambda v: v.reset_view())
