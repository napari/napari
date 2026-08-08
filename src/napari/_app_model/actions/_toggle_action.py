from __future__ import annotations

from typing import Any

from app_model import Action
from app_model.types import ToggleRule
from pydantic import PrivateAttr

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

    _attribute_path_parts: list[str] = PrivateAttr(default_factory=list)

    def __init__(
        self,
        *,
        id: str,  # noqa: A002
        title: str,
        attribute_path: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            id=id,
            title=title,
            toggled=ToggleRule(get_current=self.get_current),
            callback=self.toggle,
            **kwargs,
        )
        self._attribute_path_parts.extend(attribute_path.split('.'))

    def get_current(self, viewer: ViewerModel) -> bool:
        """return the current value of the viewer attribute"""
        attr = viewer
        for part in self._attribute_path_parts:
            attr = getattr(attr, part)
        return attr  # type: ignore

    def toggle(self, viewer: ViewerModel) -> None:
        """toggle the viewer attribute"""
        attr = viewer
        parts = self._attribute_path_parts
        for part in parts[:-1]:
            attr = getattr(attr, part)
        setattr(attr, parts[-1], not getattr(attr, parts[-1]))
