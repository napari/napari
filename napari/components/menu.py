from typing import Callable, List, Optional

from pydantic import Field

from ..utils.events import EventedModel


class MenuItem(EventedModel):
    label: str = Field(..., description='Text of the menu item.')
    id: str = Field(
        ...,
        description='Internal identifier of the menu item, to be used when accessing programmatically.',
    )
    description: Optional[str] = Field(
        description='Description to be shown in the status bar when hovered.'
    )
    # TODO: add validation to keybinding
    keybinding: Optional[str] = Field(
        description='Keyboard shortcut which activates this item.'
    )
    enabled: bool = Field(
        True, description='Whether or not to allow this item to be clickable.'
    )


class ActionMenuItem(MenuItem):
    action: Callable[[], None] = Field(
        ..., description='Callback to trigger when the menu item is clicked.'
    )


class CheckableMenuItem(MenuItem):
    checked: bool = Field(
        ..., description='Whether the current menu item is checked.'
    )


class Menu(MenuItem):
    children: List[MenuItem] = Field(
        ...,
        description='Children menu items of this menu.',
        allow_mutation=False,
    )

    def get(self, id: str) -> Optional[MenuItem]:
        """Get a child by its id.

        Parameters
        ----------
        id : str
            Id of the menu item to fetch.

        Returns
        -------
        menu_item : MenuItem, optional
            Fetched menu item, if found.
        """
        for child in self.children:
            if child.id == id:
                return child
