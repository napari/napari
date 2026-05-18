from __future__ import annotations

from typing import Any

from app_model import Action
from app_model.types import ToggleRule

from napari.components import ViewerModel


class ViewerModelToggleAction(Action):
    """Action subclass that toggles a boolean viewer (sub)attribute on trigger.

    Parameters
    ----------
    id : str
        The command id of the action.
    title : str
        The title of the action. Prefer capital case.
    attribute_path : str
        The attribute of the viewer attribute to toggle. (e.g. 'visible')
    **kwargs
        Additional keyword arguments to pass to the Action constructor.

    Examples
    --------
    >>> action = ViewerModelToggleAction(
    ...     id='some.command.id',
    ...     title='Toggle Axis Visibility',
    ...     attribute_path='axes.visible',
    ... )
    """

    def __init__(
        self,
        *,
        id: str,  # noqa: A002
        title: str,
        attribute_path: str,
        **kwargs: Any,
    ) -> None:
        def get_current(viewer: ViewerModel) -> bool:
            """return the current value of the viewer attribute"""
            attr = viewer
            for part in attribute_path.split('.'):
                attr = getattr(attr, part)
            return attr  # type: ignore[return-value]

        def toggle(viewer: ViewerModel) -> None:
            """toggle the viewer attribute"""
            attr = viewer
            parts = attribute_path.split('.')
            for part in parts[:-1]:
                attr = getattr(attr, part)
            setattr(attr, parts[-1], not getattr(attr, parts[-1]))

        super().__init__(
            id=id,
            title=title,
            toggled=ToggleRule(get_current=get_current),
            callback=toggle,
            **kwargs,
        )
