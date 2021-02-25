import numpy as np

from .viewer_model import ViewerModel


@ViewerModel.bind_key('Control')
def reset_scroll_progress(viewer):
    """Reset dims scroll progress"""

    # on key press
    viewer.dims._scroll_progress = 0
    yield

    # on key release
    viewer.dims._scroll_progress = 0


@ViewerModel.bind_key('Control-Y')
def toggle_ndisplay(viewer):
    """Toggle ndisplay."""
    if viewer.dims.ndisplay == 3:
        viewer.dims.ndisplay = 2
    else:
        viewer.dims.ndisplay = 3


ViewerModel.bind_key('Left')(ViewerModel.dims._increment_dims_left)
ViewerModel.bind_key('Right')(ViewerModel.dims._increment_dims_right)
ViewerModel.bind_key('Alt-Up')(ViewerModel.dims._focus_up)
ViewerModel.bind_key('Alt-Down')(ViewerModel.dims._focus_down)
ViewerModel.bind_key('Control-E')(ViewerModel.dims._roll)
ViewerModel.bind_key('Control-T')(ViewerModel.dims._transpose)


@ViewerModel.bind_key('Control-Backspace')
@ViewerModel.bind_key('Control-Delete')
def remove_selected(viewer):
    """Remove selected layers."""
    viewer.layers.remove_selected()


ViewerModel.bind_key('Control-A')(ViewerModel.layers.select_all)


@ViewerModel.bind_key('Control-Shift-Backspace')
@ViewerModel.bind_key('Control-Shift-Delete')
def remove_all_layers(viewer):
    """Remove all layers."""
    viewer.layers.select_all()
    viewer.layers.remove_selected()


ViewerModel.bind_key('Up')(ViewerModel.layers.select_next)
ViewerModel.bind_key('Down')(ViewerModel.layers.select_previous)


@ViewerModel.bind_key('Shift-Up')
def also_select_layer_above(viewer):
    """Also select layer above."""
    viewer.layers.select_next(shift=True)


@ViewerModel.bind_key('Shift-Down')
def also_select_layer_below(viewer):
    """Also select layer below."""
    viewer.layers.select_previous(shift=True)


ViewerModel.bind_key('Control-R')(ViewerModel.reset_view)


@ViewerModel.bind_key('Control-G')
def toggle_grid(viewer):
    """Toggle grid mode."""
    if np.all(viewer.grid_size == (1, 1)):
        viewer.grid.enabled = True
    else:
        viewer.grid.enabled = False


ViewerModel.bind_key('V')(ViewerModel.layers.toggle_selected_visibility)
