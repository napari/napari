from __future__ import annotations

from typing import TYPE_CHECKING

from napari.components.command_palette._api import CommandPalette, get_palette
from napari.components.command_palette._components import Command

if TYPE_CHECKING:
    from app_model.expressions import Expr
    from app_model.types import CommandRule

    from napari._app_model._app import NapariApplication
    from napari._app_model.context import Context
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
        *contexts, _ = cmd.id.split(":")
        title = " > ".join(contexts)
        if menu_or_submenu.when is not None:
            when = _expr_to_callable(
                expr=menu_or_submenu.when, id=cmd.id, parent=parent
            )
        else:
            when = None
        palette.register(
            _get_callback(app, cmd),
            title=title,
            desc=cmd.title,
            when=when,
        )

    return palette


def _get_callback(app: NapariApplication, cmd: CommandRule):
    return lambda: app.commands.execute_command(cmd.id).result()


def _get_context(id: str, parent: _QtMainWindow) -> Context | None:
    from napari._app_model.context import get_context

    if id.startswith("napari:layer:"):
        ll = parent._qt_viewer.viewer.layers
        return get_context(ll)
    return None


def _expr_to_callable(expr: Expr, id: str, parent: _QtMainWindow):
    context = _get_context(id, parent)

    def _eval():
        expr.eval(context)

    return _eval
