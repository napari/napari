from typing import TYPE_CHECKING

from app_model.types import Action, ToggleRule

if TYPE_CHECKING:
    from qtpy.QtWidgets import QAction

    from ...utils.events import EventEmitter
    from ...viewer import Viewer
    from ..constants import CommandId


class ViewerToggleAction(Action):
    """Action subclass that toggles a boolean viewer (sub)attribute on trigger.

    Parameters
    ----------
    id : CommandId
        The id of the action.
    title : str
        The title of the action.
    viewer_attribute : str
        The attribute of the viewer to toggle. (e.g. 'axes')
    sub_attribute : str
        The attribute of the viewer attribute to toggle. (e.g. 'visible')
    **kwargs
        Additional keyword arguments to pass to the Action constructor.

    Examples
    --------
    >>> action = ViewerToggleAction(
    ...     id='some.command.id',
    ...     title='Toggle Axis Visibility',
    ...     viewer_attribute='axes',
    ...     sub_attribute='visible',
    ... )
    """

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
            """will be called to toggle the attribute when the action is triggered"""
            attr = getattr(viewer, viewer_attribute)
            current = getattr(attr, sub_attribute)
            setattr(attr, sub_attribute, not current)

        def initialize(viewer: 'Viewer'):
            """will be called to initialize value of the action on creation"""
            attr = getattr(viewer, viewer_attribute)
            return getattr(attr, sub_attribute)

        def connect(action: 'QAction', viewer: 'Viewer'):
            """will be called upon action creation, we connect to events here."""
            attr = getattr(viewer, viewer_attribute)
            emitter: EventEmitter = getattr(attr.events, sub_attribute)

            @emitter.connect
            def _setchecked(e):
                action.setChecked(e.value if hasattr(e, 'value') else e)

            action.destroyed.connect(lambda: emitter.disconnect(_setchecked))

        super().__init__(
            id=id,
            title=title,
            toggled=ToggleRule(
                initialize=initialize, experimental_connect=connect
            ),
            callback=toggle,
            **kwargs,
        )
