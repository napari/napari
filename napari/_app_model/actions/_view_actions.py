"""Actions related to the 'View' menu that do not require Qt.

View actions that do require Qt should go in
`napari/_qt/_qapp_model/qactions/_view.py`.
"""

from typing import List

from app_model.types import Action, ToggleRule

from ...settings import get_settings
from ..constants import CommandId, MenuId
from ._toggle_action import ViewerToggleAction

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


VIEW_ACTIONS.extend(
    [
        # TODO: this could be made into a toggle setting Action subclass
        # using a similar pattern to the above ViewerToggleAction classes
        Action(
            id=CommandId.TOGGLE_LAYER_TOOLTIPS,
            title=CommandId.TOGGLE_LAYER_TOOLTIPS.title,
            menus=[
                {'id': MenuId.MENUBAR_VIEW, 'group': '1_render', 'order': 10}
            ],
            callback=_tooltip_visibility_toggle,
            toggled=ToggleRule(get_current=_get_current_tooltip_visibility),
        ),
    ]
)
