"""Actions related to the 'View' menu that do not require Qt.

View actions that do require Qt should go in
`napari/_qt/_qapp_model/qactions/_view.py`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from app_model.types import Action, ToggleRule

from napari._app_model.actions._toggle_action import ViewerToggleAction
from napari._app_model.constants import CommandId, MenuId
from napari.settings import get_settings

if TYPE_CHECKING:
    from napari.viewer import Viewer


VIEW_ACTIONS: List[Action] = []

for cmd, viewer_attr, sub_attr in (
    (CommandId.TOGGLE_VIEWER_AXES, 'axes', 'visible'),
    (CommandId.TOGGLE_VIEWER_AXES_COLORED, 'axes', 'colored'),
    (CommandId.TOGGLE_VIEWER_AXES_LABELS, 'axes', 'labels'),
    (CommandId.TOGGLE_VIEWER_AXES_DASHED, 'axes', 'dashed'),
    (CommandId.TOGGLE_VIEWER_AXES_ARROWS, 'axes', 'arrows'),
    (CommandId.TOGGLE_VIEWER_SCALE_BAR, 'scale_bar', 'visible'),
    (CommandId.TOGGLE_VIEWER_SCALE_BAR_COLORED, 'scale_bar', 'colored'),
    (CommandId.TOGGLE_VIEWER_SCALE_BAR_TICKS, 'scale_bar', 'ticks'),
):
    menu = MenuId.VIEW_AXES if viewer_attr == 'axes' else MenuId.VIEW_SCALEBAR
    VIEW_ACTIONS.append(
        ViewerToggleAction(
            id=cmd,
            title=cmd.title,
            viewer_attribute=viewer_attr,
            sub_attribute=sub_attr,
            menus=[{'id': menu}],
        )
    )


def _tooltip_visibility_toggle():
    settings = get_settings().appearance
    settings.layer_tooltip_visibility = not settings.layer_tooltip_visibility


# this can be generalised for all boolean settings, similar to `ViewerToggleAction`
def _get_current_tooltip_visibility():
    return get_settings().appearance.layer_tooltip_visibility


VIEW_ACTIONS.append(
    # TODO: this could be made into a toggle setting Action subclass
    # using a similar pattern to the above ViewerToggleAction classes
    Action(
        id=CommandId.TOGGLE_LAYER_TOOLTIPS,
        title=CommandId.TOGGLE_LAYER_TOOLTIPS.title,
        menus=[{'id': MenuId.MENUBAR_VIEW, 'group': '1_render', 'order': 10}],
        callback=_tooltip_visibility_toggle,
        toggled=ToggleRule(get_current=_get_current_tooltip_visibility),
    ),
)


def _ndisplay_toggle(viewer: Viewer):
    viewer.dims.ndisplay = 2 + (viewer.dims.ndisplay == 2)


def _get_current_ndisplay_is_3D(viewer: Viewer):
    return viewer.dims.ndisplay == 3


# actions ported to app_model from components/_viewer_key_bindings
VIEW_ACTIONS.extend(
    [
        Action(
            id=CommandId.TOGGLE_VIEWER_NDISPLAY,
            title=CommandId.TOGGLE_VIEWER_NDISPLAY.title,
            menus=[
                {'id': MenuId.MENUBAR_VIEW, 'group': '1_render', 'order': 0}
            ],
            callback=_ndisplay_toggle,
            toggled=ToggleRule(get_current=_get_current_ndisplay_is_3D),
        ),
        ViewerToggleAction(
            id=CommandId.VIEWER_TOGGLE_GRID,
            title=CommandId.VIEWER_TOGGLE_GRID.title,
            viewer_attribute="grid",
            sub_attribute="enabled",
            menus=[
                {'id': MenuId.MENUBAR_VIEW, 'group': '1_render', 'order': 0},
            ],
        ),
    ],
)
