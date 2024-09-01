from __future__ import annotations

from typing import TYPE_CHECKING

from napari.components.command_palette._api import CommandPalette, get_palette
from napari.components.command_palette._components import Command

if TYPE_CHECKING:
    from app_model.types import CommandRule

    from napari._app_model._app import NapariApplication
    from napari._qt.qt_main_window import _QtMainWindow


__all__ = [
    "Command",
    "get_palette",
    "CommandPalette",
    "get_napari_command_palette",
    "create_napari_command_palette",
]


def get_napari_command_palette():
    """Get the command palette for the current napari app name."""
    from napari._app_model import get_app

    app = get_app()
    return get_palette(app.name)


def create_napari_command_palette(parent: _QtMainWindow) -> CommandPalette:
    """Get the napari command palette and initialize commands."""
    from napari._app_model import get_app

    app = get_app()
    palette = get_palette(app.name)

    all_menus = app.menus.get_menu(app.menus.COMMAND_PALETTE_ID)
    for menu_or_submenu in all_menus:
        cmd = menu_or_submenu.command
        sep = ":" if ":" in cmd.id else "."
        *contexts, _ = cmd.id.split(sep)
        title = " > ".join(contexts)
        if menu_or_submenu.when is not None:
            enabled = menu_or_submenu.when
        else:
            enabled = None
        palette.register(
            _get_callback(app, cmd),
            title=title,
            desc=cmd.title,
            enablement=enabled,
        )
    return palette


def _get_callback(app: NapariApplication, cmd: CommandRule):
    return lambda: app.commands.execute_command(cmd.id).result()
