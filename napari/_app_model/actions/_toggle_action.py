from __future__ import annotations

from typing import TYPE_CHECKING

from app_model.types import Action, ToggleRule

if TYPE_CHECKING:
    from ...viewer import Viewer


class ViewerToggleAction(Action):
    """Action subclass that toggles a boolean viewer (sub)attribute on trigger.

    Parameters
    ----------
    id : str
        The command id of the action.
    title : str
        The title of the action. Prefer capital case.
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
        id: str,
        title: str,
        viewer_attribute: str,
        sub_attribute: str,
        **kwargs,
    ):
        def get_current(viewer: Viewer):
            """return the current value of the viewer attribute"""
            attr = getattr(viewer, viewer_attribute)
            return getattr(attr, sub_attribute)

        def toggle(viewer: Viewer):
            """toggle the viewer attribute"""
            attr = getattr(viewer, viewer_attribute)
            setattr(attr, sub_attribute, not getattr(attr, sub_attribute))

        super().__init__(
            id=id,
            title=title,
            toggled=ToggleRule(get_current=get_current),
            callback=toggle,
            **kwargs,
        )
