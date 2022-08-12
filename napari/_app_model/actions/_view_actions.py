from typing import TYPE_CHECKING, List

from app_model.types import Action, ToggleRule

from ..constants import CommandId, MenuId

if TYPE_CHECKING:
    from ...viewer import Viewer


class ViewerToggleAction(Action):
    def __init__(
        self,
        viewer_attribute: str,
        property: str,
        **kwargs,
    ):
        def toggle(viewer: 'Viewer'):
            attr = getattr(viewer, viewer_attribute)
            current = getattr(attr, property)
            setattr(attr, property, not current)

        def initialize(viewer: 'Viewer'):
            attr = getattr(viewer, viewer_attribute)
            return getattr(attr, property)

        def connect(action, viewer: 'Viewer'):
            attr = getattr(viewer, viewer_attribute)
            emitter = getattr(attr.events, property)

            @emitter.connect
            def _setchecked(e):
                action.setChecked(e.value if hasattr(e, 'value') else e)

            action.destroyed.connect(lambda: emitter.disconnect(_setchecked))

        rule = ToggleRule(initialize=initialize, connect=connect)
        super().__init__(toggled=rule, callback=toggle, **kwargs)


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
            property=prop,
            menus=[{'id': menu}],
        )
    )
