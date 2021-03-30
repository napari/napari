from ..utils.key_bindings import action_manager
from ..utils.settings import SETTINGS
from ..utils.theme import available_themes
from ..utils.translations import trans
from .viewer_model import ViewerModel

action_manager.register_action(
    'reset_view',
    lambda viewer: ViewerModel.reset_view(viewer),
    trans._("Reset view to original state."),
    ViewerModel,
)


@ViewerModel.bind_key('Control')
def reset_scroll_progress(viewer):

    # on key press
    viewer.dims._scroll_progress = 0
    yield

    # on key release
    viewer.dims._scroll_progress = 0


reset_scroll_progress.__doc__ = trans._("Reset dims scroll progress")


@ViewerModel.register_action(description=trans._("Toggle ndisplay."))
def toggle_ndisplay(viewer):
    if viewer.dims.ndisplay == 3:
        viewer.dims.ndisplay = 2
    else:
        viewer.dims.ndisplay = 3


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


@ViewerModel.register_action(
    description=trans._("Increment dimensions slider to the left.")
)
def increment_dims_left(viewer):
    viewer.dims._increment_dims_left()


@ViewerModel.register_action(
    description=trans._("Increment dimensions slider to the right.")
)
def increment_dims_right(viewer):
    """Increment dimensions slider to the right."""
    viewer.dims._increment_dims_right()


@ViewerModel.register_action(
    description=trans._("Move focus of dimensions slider up.")
)
def focus_axes_up(viewer):
    """Move focus of dimensions slider up."""
    viewer.dims._focus_up()


@ViewerModel.register_action(
    description=trans._("Move focus of dimensions slider down.")
)
def focus_axes_down(viewer):
    """Move focus of dimensions slider down."""
    viewer.dims._focus_down()


@ViewerModel.register_action(
    description=trans._(
        "Change order of the visible axes, e.g. [0, 1, 2] -> [2, 0, 1]."
    )
)
def roll_axes(viewer):
    """Roll dimensions order for display"""
    viewer.dims._roll()


@ViewerModel.register_action(
    description=trans._(
        "Transpose order of the last two visible axes, e.g. [0, 1] -> [1, 0]."
    )
)
def transpose_axes(viewer):
    """Transpose order of the last two visible axes, e.g. [0, 1] -> [1, 0]."""
    viewer.dims._transpose()


@ViewerModel.register_action(description=trans._("Remove selected layers."))
def remove_selected(viewer):
    """Remove selected layers."""
    viewer.layers.remove_selected()


@ViewerModel.register_action(description=trans._("Selected all layers."))
def select_all(viewer):
    """Selected all layers."""
    viewer.layers.select_all()


@ViewerModel.register_action(description=trans._("Remove all layers."))
def remove_all_layers(viewer):
    """Remove all layers."""
    viewer.layers.clear()


@ViewerModel.register_action(description=trans._("Select layer above."))
def select_layer_above(viewer):
    """Select layer above."""
    viewer.layers.select_next()


@ViewerModel.register_action(description=trans._("Select layer below."))
def select_layer_below(viewer):
    """Select layer below."""
    viewer.layers.select_previous()


@ViewerModel.register_action(description=trans._("Also select layer above."))
def also_select_layer_above(viewer):
    """Also select layer above."""
    viewer.layers.select_next(shift=True)


@ViewerModel.register_action(description=trans._("Also select layer below."))
def also_select_layer_below(viewer):
    """Also select layer below."""
    viewer.layers.select_previous(shift=True)


@ViewerModel.register_action(description=trans._("Toggle grid mode."))
def toggle_grid(viewer):
    """Toggle grid mode."""
    viewer.grid.enabled = not viewer.grid.enabled


@ViewerModel.register_action(
    description=trans._("Toggle visibility of selected layers")
)
def toggle_selected_visibility(viewer):
    viewer.layers.toggle_selected_visibility()
