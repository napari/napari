from __future__ import annotations

from napari.components.command_palette._api import CommandPalette, get_palette
from napari.components.command_palette._components import Command

__all__ = [
    'Command',
    'get_palette',
    'CommandPalette',
    'create_napari_command_palette',
]


def create_napari_command_palette() -> CommandPalette:
    """Get the napari command palette and initialize commands."""
    from napari._app_model import get_app

    app = get_app()
    palette = get_palette(app.name)

    all_menus = app.menus.get_menu(app.menus.COMMAND_PALETTE_ID)
    for menu_or_submenu in all_menus:
        cmd = menu_or_submenu.command
        palette.register(cmd)
    return palette
