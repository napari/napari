from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, List, Optional

from npe2 import plugin_manager
from npe2.manifest import contributions

from app_model.registries import CommandsRegistry, KeybindingsRegistry, MenuRegistry
from app_model.types import (
    CommandRule,
    MenuItem,
    SubmenuItem,
    _RegisteredCommand,
)

if TYPE_CHECKING:
    from ._plugin_aware_registries import MenuOrSubmenu


def npe2_command_to_rule(cmd_id: str) -> CommandRule:
    cmd_contrib = plugin_manager.get_command(cmd_id)
    return CommandRule(
        id=cmd_contrib.id,
        title=cmd_contrib.title,
        category=cmd_contrib.category,
        icon=cmd_contrib.icon,
        enablement=cmd_contrib.enablement,
        short_title=cmd_contrib.short_title,
    )


def convert_npe2_menu_item(npe2_item: contributions.MenuItem) -> MenuItem:
    group, _, order = (npe2_item.group or '').partition("@")
    try:
        order = float(order)
    except ValueError:
        order = None

    if isinstance(npe2_item, contributions.MenuCommand):
        return MenuItem(
            command=npe2_command_to_rule(npe2_item.command),
            when=npe2_item.when,
            group=group,
            order=order,
        )

    subm = plugin_manager.get_submenu(npe2_item.submenu)
    return SubmenuItem(
        submenu=subm.id,
        title=subm.label,
        icon=subm.icon,
        when=npe2_item.when,
        group=group,
        order=order,
    )


# IMPROVE
class PluginAwareCommandsRegistry(CommandsRegistry):
    __instance: Optional[PluginAwareCommandsRegistry] = None

    @classmethod
    def instance(cls) -> PluginAwareCommandsRegistry:
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance

    def __getitem__(self, id: CommandId) -> List[_RegisteredCommand]:
        try:
            self._commands[id]
        except KeyError:
            cmd = plugin_manager.get_command(id)
            return [_RegisteredCommand(id, cmd.exec, cmd.title)]


class PluginAwareKeybindingsRegistry(KeybindingsRegistry):
    __instance: Optional[PluginAwareKeybindingsRegistry] = None

    @classmethod
    def instance(cls) -> PluginAwareKeybindingsRegistry:
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance


class PluginAwareMenuRegistry(MenuRegistry):
    __instance: Optional[PluginAwareMenuRegistry] = None

    @classmethod
    def instance(cls) -> PluginAwareMenuRegistry:
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance

    def get_menu(
        self, menu_id: str, include_plugins: bool = True
    ) -> List[MenuOrSubmenu]:
        items = self._menu_items.get(menu_id, [])
        if include_plugins:
            menu_contribs = plugin_manager.iter_menu(menu_id)
            items.extend([convert_npe2_menu_item(i) for i in menu_contribs])
        if not items:
            raise KeyError("No menu found with id: {menu_id}")
        return items

    def iter_menu_groups(
        self, menu_id: str, include_plugins: bool = True
    ) -> Iterator[List[MenuOrSubmenu]]:
        menu_items = self.get_menu(menu_id, include_plugins)
        yield from self._sorted_groups(menu_items)


commands_registry = PluginAwareCommandsRegistry.instance()
menu_registry = PluginAwareMenuRegistry.instance()
keybindings_registry = PluginAwareKeybindingsRegistry.instance()
