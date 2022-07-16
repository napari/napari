from typing import TYPE_CHECKING, List

from app_model.types import Action

from ..constants import CommandId, MenuId

if TYPE_CHECKING:
    from ...viewer import Viewer


def _toggler(viewer_attribute: str, property: str):
    def _callback(viewer: 'Viewer'):
        sub_model = getattr(viewer, viewer_attribute)
        current = getattr(sub_model, property)
        setattr(sub_model, property, not current)

    return _callback


VIEW_ACTIONS: List[Action] = []

for cmd, viewer_attr, prop in (
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
        Action(
            id=cmd,
            title=cmd.title,
            callback=_toggler(viewer_attr, prop),
            menus=[{'id': menu}],
            # FIXME: this is a hack that will work if it is only toggled via
            # the menu, but not if it is controlled programmatically. It will
            # also be wrong if the attribute starts out toggled.
            # we need a proper context key for this.
            toggled='True',
        )
    )
