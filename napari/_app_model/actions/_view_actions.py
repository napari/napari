from typing import TYPE_CHECKING, List

from app_model.types import Action, ToggleRule

from ..constants import CommandId, MenuId

if TYPE_CHECKING:
    from qtpy.QtWidgets import QAction

    from ...utils.events import EventEmitter
    from ...viewer import Viewer


class ViewerToggleAction(Action):
    def __init__(
        self,
        *,
        id: CommandId,
        title: str,
        viewer_attribute: str,
        sub_attribute: str,
        **kwargs,
    ):
        def toggle(viewer: 'Viewer'):
            attr = getattr(viewer, viewer_attribute)
            current = getattr(attr, sub_attribute)
            setattr(attr, sub_attribute, not current)

        def initialize(viewer: 'Viewer'):
            attr = getattr(viewer, viewer_attribute)
            return getattr(attr, sub_attribute)

        def connect(action: 'QAction', viewer: 'Viewer'):
            attr = getattr(viewer, viewer_attribute)
            emitter: EventEmitter = getattr(attr.events, sub_attribute)

            @emitter.connect
            def _setchecked(e):
                action.setChecked(e.value if hasattr(e, 'value') else e)

            action.destroyed.connect(lambda: emitter.disconnect(_setchecked))

        rule = ToggleRule(initialize=initialize, experimental_connect=connect)
        super().__init__(
            id=id,
            title=title,
            toggled=rule,
            callback=toggle,
            **kwargs,
        )


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
        ViewerToggleAction(
            id=cmd,
            title=cmd.title,
            viewer_attribute=viewer_attr,
            sub_attribute=prop,
            menus=[{'id': menu}],
        )
    )
