from __future__ import annotations

import contextlib
from typing import List, Optional, Union

from app_model.registries import CommandsRegistry, MenusRegistry
from app_model.registries._commands_reg import _RegisteredCommand  # FIXME
from app_model.types import CommandRule, MenuItem, MenuOrSubmenu, SubmenuItem
from npe2 import plugin_manager
from npe2.manifest import contributions

CommandId = str


def _partition_group_order(
    group: Optional[str],
) -> dict[str, Union[str, float, None]]:
    """Split a npe2 group string into a dict of group and order.

    Examples
    --------
    >>> _partition_group_order("my_group@4")
    {"group": "my_group", "order": 4}
    >>> _partition_group_order("my_group")
    {"group": "my_group", "order": None}
    >>> _partition_group_order("")
    {"group": None, "order": None}
    """
    group, _, _order = (group or '').partition("@")
    try:
        order: Optional[float] = float(_order)
    except ValueError:
        order = None
    return {'group': group or None, 'order': order}


def _npe2_command_to_app_model(
    cmd: contributions.CommandContribution,
) -> CommandRule:
    """Convert a npe2 command contribution to an app_model command rule."""
    return CommandRule(
        id=cmd.id,
        title=cmd.title,
        category=cmd.category,
        icon=cmd.icon,
        enablement=cmd.enablement,
        short_title=cmd.short_title,
    )


def _npe2_submenu_to_app_model(subm: contributions.Submenu) -> SubmenuItem:
    """Convert a npe2 submenu contribution to an app_model SubmenuItem."""
    contrib = plugin_manager.get_submenu(subm.submenu)
    return SubmenuItem(
        submenu=contrib.id,
        title=contrib.label,
        icon=contrib.icon,
        when=subm.when,
        **_partition_group_order(subm.group),
        # enablement= ??  npe2 doesn't have this, but app_model does
    )


def _npe2_menu_cmd_to_app_model(
    menu_cmd: contributions.MenuCommand,
) -> MenuItem:
    """Convert a npe2 menu command contribution to an app_model MenuItem."""
    contrib = plugin_manager.get_command(menu_cmd.command)
    return MenuItem(
        command=_npe2_command_to_app_model(contrib),
        when=menu_cmd.when,
        **_partition_group_order(menu_cmd.group),
    )


def _npe2_menu_to_app_model(
    npe2_item: contributions.MenuItem,
) -> MenuOrSubmenu:
    """Convert a npe2 MenuItem to an app_model MenuOrSubmenu.

    just picks the appropriate function given the type of the npe2 item.
    """
    if isinstance(npe2_item, contributions.MenuCommand):
        return _npe2_menu_cmd_to_app_model(npe2_item)
    elif isinstance(npe2_item, contributions.Submenu):
        return _npe2_submenu_to_app_model(npe2_item)
    raise TypeError(f"Unknown npe2 MenuItem type: {type(npe2_item)}")


class PluginAwareCommandsRegistry(CommandsRegistry):
    def __getitem__(self, id: CommandId) -> _RegisteredCommand:
        with contextlib.suppress(KeyError):
            cmd = plugin_manager.get_command(id)
            # FIXME: this will probably not inject properly
            # also, this should be cached somehow
            return _RegisteredCommand(
                id=id, callback=cmd.exec, title=cmd.title
            )
        return super().__getitem__(id)


class PluginAwareMenusRegistry(MenusRegistry):
    def __contains__(self, id: object) -> bool:
        return any(
            True for _ in plugin_manager.iter_menu(id)
        ) or super().__contains__(id)

    def get_menu(
        self, menu_id: str, include_plugins: bool = True
    ) -> List[MenuOrSubmenu]:
        items = self._menu_items.get(menu_id, [])
        if include_plugins:
            items.extend(
                _npe2_menu_to_app_model(i)
                for i in plugin_manager.iter_menu(menu_id)
            )
        if not items:
            raise KeyError("No menu found with id: {menu_id}")
        return items
