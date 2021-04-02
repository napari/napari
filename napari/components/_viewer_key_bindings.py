from ..utils.settings import SETTINGS
from ..utils.theme import available_themes
from ..utils.translations import trans
from .viewer_model import ViewerModel


@ViewerModel.bind_key('Control')
def reset_scroll_progress(viewer):

    # on key press
    viewer.dims._scroll_progress = 0
    yield

    # on key release
    viewer.dims._scroll_progress = 0


reset_scroll_progress.__doc__ = trans._("Reset dims scroll progress")


@ViewerModel.bind_key('Control-Y')
def toggle_ndisplay(viewer):
    """Toggle ndisplay."""
    if viewer.dims.ndisplay == 3:
        viewer.dims.ndisplay = 2
    else:
        viewer.dims.ndisplay = 3


toggle_ndisplay.__doc__ = trans._("Toggle ndisplay.")


@ViewerModel.bind_key('Control-Shift-T')
def toggle_theme(viewer):
    """Toggle theme for viewer"""
    themes = available_themes()
    current_theme = SETTINGS.appearance.theme
    idx = themes.index(current_theme)
    idx += 1
    if idx == len(themes):
        idx = 0

    SETTINGS.appearance.theme = themes[idx]


@ViewerModel.bind_key('Left')
def increment_dims_left(viewer):
    """Increment dimensions slider to the left."""
    viewer.dims._increment_dims_left()


increment_dims_left.__doc__ = trans._(
    "Increment dimensions slider to the left."
)


@ViewerModel.bind_key('Right')
def increment_dims_right(viewer):
    """Increment dimensions slider to the right."""
    viewer.dims._increment_dims_right()


increment_dims_right.__doc__ = trans._(
    "Increment dimensions slider to the right."
)


@ViewerModel.bind_key('Alt-Up')
def focus_axes_up(viewer):
    """Move focus of dimensions slider up."""
    viewer.dims._focus_up()


focus_axes_up.__doc__ = trans._("Move focus of dimensions slider up.")


@ViewerModel.bind_key('Alt-Down')
def focus_axes_down(viewer):
    """Move focus of dimensions slider down."""
    viewer.dims._focus_down()


focus_axes_down.__doc__ = trans._("Move focus of dimensions slider down.")


@ViewerModel.bind_key('Control-E')
def roll_axes(viewer):
    """Change order of the visible axes, e.g. [0, 1, 2] -> [2, 0, 1]."""
    viewer.dims._roll()


roll_axes.__doc__ = trans._(
    "Change order of the visible axes, e.g. [0, 1, 2] -> [2, 0, 1]."
)


@ViewerModel.bind_key('Control-T')
def transpose_axes(viewer):
    """Transpose order of the last two visible axes, e.g. [0, 1] -> [1, 0]."""
    viewer.dims._transpose()


transpose_axes.__doc__ = trans._(
    "Transpose order of the last two visible axes, e.g. [0, 1] -> [1, 0]."
)


@ViewerModel.bind_key('Control-R')
def reset_view(viewer):
    """Reset view to original state."""
    viewer.reset_view()


reset_view.__doc__ = trans._("Reset view to original state.")


@ViewerModel.bind_key('Control-G')
def toggle_grid(viewer):
    """Toggle grid mode."""
    viewer.grid.enabled = not viewer.grid.enabled


toggle_grid.__doc__ = trans._("Toggle grid mode.")


@ViewerModel.bind_key('V')
def toggle_selected_visibility(viewer):
    """Toggle visibility of selected layers"""
    viewer.layers.toggle_selected_visibility()


toggle_selected_visibility.__doc__ = trans._(
    "Toggle visibility of selected layers"
)
