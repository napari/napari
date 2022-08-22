from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from app_model.types import Action, ToggleRule

if TYPE_CHECKING:
    from ...viewer import Viewer


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
        id: str,
        title: str,
        viewer_attribute: str,
        sub_attribute: str,
        **kwargs,
    ):
        # these functions take an optional viewer, since they may be called
        # during tests in the absence of a Viewer window (in which case Viewer is None)
        def get_current(viewer: Optional[Viewer] = None) -> bool:
            """return the current value of the viewer attribute"""
            if viewer is None:
                return False
            attr = getattr(viewer, viewer_attribute)
            return getattr(attr, sub_attribute)

        def toggle(viewer: Optional[Viewer] = None) -> None:
            """toggle the viewer attribute"""
            if viewer is None:
                return
            attr = getattr(viewer, viewer_attribute)
            setattr(attr, sub_attribute, not getattr(attr, sub_attribute))

        super().__init__(
            id=id,
            title=title,
            toggled=ToggleRule(get_current=get_current),
            callback=toggle,
            **kwargs,
        )
