from typing import Callable, List, Optional

from ..utils.events import EventedModel


class MenuItem(EventedModel):
    label: str
    id: str
    description: Optional[str]
    # TODO: add validation to keybinding
    keybinding: Optional[str]


class ActionMenuItem(MenuItem):
    action: Callable[[], None]


class CheckableMenuItem(MenuItem):
    checked: bool


class Menu(MenuItem):
    # should this be an EventedList?
    children: List[MenuItem]
