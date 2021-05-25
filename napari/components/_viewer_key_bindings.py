from ..utils.action_manager import action_manager
from ..utils.settings import SETTINGS
from ..utils.theme import available_themes
from ..utils.translations import trans
from .viewer_model import ViewerModel


def register_viewer_action(description):
    """
    Convenient decorator to register an action with the current ViewerModel

    It will use the function name as the action name. We force the description
    to be given instead of function docstring for translation purpose.
    """

    def _inner(func):
        name = 'napari:' + func.__name__
        action_manager.register_action(
            name=name,
            command=func,
            description=description,
            keymapprovider=ViewerModel,
        )
        return func

    return _inner


@register_viewer_action(trans._("Reset scroll."))
def reset_scroll_progress(viewer):

    # on key press
    viewer.dims._scroll_progress = 0
    yield

    # on key release
    viewer.dims._scroll_progress = 0


reset_scroll_progress.__doc__ = trans._("Reset dims scroll progress")


@register_viewer_action(trans._("Toggle ndisplay."))
def toggle_ndisplay(viewer):
    if viewer.dims.ndisplay == 3:
        viewer.dims.ndisplay = 2
    else:
        viewer.dims.ndisplay = 3


# Making this an action makes vispy really unhappy during the tests
# on mac only with:
# ```
# RuntimeError: wrapped C/C++ object of type CanvasBackendDesktop has been deleted
# ```
@register_viewer_action(trans._("Toggle theme."))
def toggle_theme(viewer):
    """Toggle theme for viewer"""
    themes = available_themes()
    current_theme = SETTINGS.appearance.theme
    idx = themes.index(current_theme)
    idx += 1
    if idx == len(themes):
        idx = 0

    SETTINGS.appearance.theme = themes[idx]


@register_viewer_action(trans._("Reset view to original state."))
def reset_view(viewer):
    viewer.reset_view()


@register_viewer_action(trans._("Increment dimensions slider to the left."))
def increment_dims_left(viewer):
    viewer.dims._increment_dims_left()


@register_viewer_action(trans._("Increment dimensions slider to the right."))
def increment_dims_right(viewer):
    viewer.dims._increment_dims_right()


@register_viewer_action(trans._("Move focus of dimensions slider up."))
def focus_axes_up(viewer):
    viewer.dims._focus_up()


@register_viewer_action(trans._("Move focus of dimensions slider down."))
def focus_axes_down(viewer):
    viewer.dims._focus_down()


@register_viewer_action(
    trans._("Change order of the visible axes, e.g. [0, 1, 2] -> [2, 0, 1]."),
)
def roll_axes(viewer):
    viewer.dims._roll()


@register_viewer_action(
    trans._(
        "Transpose order of the last two visible axes, e.g. [0, 1] -> [1, 0]."
    ),
)
def transpose_axes(viewer):
    viewer.dims._transpose()


@register_viewer_action(trans._("Remove selected layers."))
def remove_selected(viewer):
    viewer.layers.remove_selected()


@register_viewer_action(trans._("Selected all layers."))
def select_all(viewer):
    viewer.layers.select_all()


@register_viewer_action(trans._("Remove all layers."))
def remove_all_layers(viewer):
    viewer.layers.clear()


@register_viewer_action(trans._("Select layer above."))
def select_layer_above(viewer):
    viewer.layers.select_next()


@register_viewer_action(trans._("Select layer below."))
def select_layer_below(viewer):
    viewer.layers.select_previous()


@register_viewer_action(trans._("Also select layer above."))
def also_select_layer_above(viewer):
    viewer.layers.select_next(shift=True)


@register_viewer_action(trans._("Also select layer below."))
def also_select_layer_below(viewer):
    viewer.layers.select_previous(shift=True)


@register_viewer_action(trans._("Toggle grid mode."))
def toggle_grid(viewer):
    viewer.grid.enabled = not viewer.grid.enabled


@register_viewer_action(trans._("Toggle visibility of selected layers"))
def toggle_selected_visibility(viewer):
    viewer.layers.toggle_selected_visibility()
