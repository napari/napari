from typing import Union

from app_model.types import Action, MenuItem, SubmenuItem

from napari._app_model import get_app
from napari._app_model.constants._menus import _EMPTY_MENU_DUMMY_ID

MenuOrSubmenu = Union[MenuItem, SubmenuItem]


def contains_dummy_action(menu_items: list[MenuOrSubmenu]) -> bool:
    """Returns True if one of the menu_items is the dummy action, otherwise False.

    Parameters
    ----------
    menu_items : list[MenuOrSubmenu]
        menu items belonging to a given menu

    Returns
    -------
    bool
        True if menu_items contains dummy item otherewise false
    """
    for item in menu_items:
        if (
            hasattr(item, 'command')
            and item.command.id == _EMPTY_MENU_DUMMY_ID
        ):
            return True
    return False


def is_empty_menu(menu_id: str) -> bool:
    """Returns True if the given menu_id is empty, otherwise False

    Parameters
    ----------
    menu_id : str
        id of the menu to check

    Returns
    -------
    bool
        True if the given menu_id is empty, otherwise False
    """
    app = get_app()
    if menu_id not in app.menus:
        return True
    if len(app.menus.get_menu(menu_id)) == 0:
        return True
    return False


def no_op():
    """Fully qualified no-op to use for dummy actions."""


def get_dummy_action(menus_list: list[dict]) -> Action:
    """Returns a dummy action to be used for all given menus.

    Parameters
    ----------
    menus_list : list[dict]
        list of dictionaries with menu attributes where the dummy will be added

    Returns
    -------
    Action
        dummy action with the dummy action ID and a no-op callback
    """
    return Action(
        id=_EMPTY_MENU_DUMMY_ID,
        title='Empty',
        callback=no_op,
        menus=menus_list,
    )
