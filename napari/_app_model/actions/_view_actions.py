"""Actions related to the 'View' menu that do not require Qt.

View actions that do require Qt should go in
`napari/_qt/_qapp_model/qactions/_view.py`.
"""

from typing import List

from app_model.types import Action, ToggleRule

from napari._app_model.actions._toggle_action import ViewerToggleAction
from napari._app_model.constants import MenuGroup, MenuId
from napari.settings import get_settings
from napari.utils.translations import trans

VIEW_ACTIONS: List[Action] = []
MENUID_DICT = {'axes': MenuId.VIEW_AXES, 'scale_bar': MenuId.VIEW_SCALEBAR}

for cmd, cmd_title, viewer_attr, sub_attr in (
    (
        'napari.window.view.toggle_viewer_axes',
        trans._('Axes Visible'),
        'axes',
        'visible',
    ),
    (
        'napari.window.view.toggle_viewer_axes_colored',
        trans._('Axes Colored'),
        'axes',
        'colored',
    ),
    (
        'napari.window.view.toggle_viewer_axes_labels',
        trans._('Axes Labels'),
        'axes',
        'labels',
    ),
    (
        'napari.window.view.toggle_viewer_axesdashed',
        trans._('Axes Dashed'),
        'axes',
        'dashed',
    ),
    (
        'napari.window.view.toggle_viewer_axes_arrows',
        trans._('Axes Arrows'),
        'axes',
        'arrows',
    ),
    (
        'napari.window.view.toggle_viewer_scale_bar',
        trans._('Scale Bar Visible'),
        'scale_bar',
        'visible',
    ),
    (
        'napari.window.view.toggle_viewer_scale_bar_colored',
        trans._('Scale Bar Colored'),
        'scale_bar',
        'colored',
    ),
    (
        'napari.window.view.toggle_viewer_scale_bar_ticks',
        trans._('Scale Bar Ticks'),
        'scale_bar',
        'ticks',
    ),
):
    VIEW_ACTIONS.append(
        ViewerToggleAction(
            id=cmd,
            title=cmd_title,
            viewer_attribute=viewer_attr,
            sub_attribute=sub_attr,
            menus=[{'id': MENUID_DICT[viewer_attr]}],
        )
    )


def _tooltip_visibility_toggle() -> None:
    settings = get_settings().appearance
    settings.layer_tooltip_visibility = not settings.layer_tooltip_visibility


def _get_current_tooltip_visibility() -> bool:
    return get_settings().appearance.layer_tooltip_visibility


VIEW_ACTIONS.extend(
    [
        # TODO: this could be made into a toggle setting Action subclass
        # using a similar pattern to the above ViewerToggleAction classes
        Action(
            id='napari.window.view.toggle_layer_tooltips',
            title=trans._('Toggle Layer Tooltips'),
            menus=[
                {
                    'id': MenuId.MENUBAR_VIEW,
                    'group': MenuGroup.RENDER,
                    'order': 10,
                }
            ],
            callback=_tooltip_visibility_toggle,
            toggled=ToggleRule(get_current=_get_current_tooltip_visibility),
        ),
    ]
)
