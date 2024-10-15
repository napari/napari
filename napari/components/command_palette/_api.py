from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

from app_model.expressions import Constant

from napari.components.command_palette._components import Command

if TYPE_CHECKING:
    from app_model.types import CommandRule
    from qtpy import QtWidgets as QtW

    from napari._qt.widgets.qt_command_palette import QCommandPalette

    _WVDict = weakref.WeakValueDictionary[int, QtW.QWidget]


class CommandPalette:
    """The command palette interface."""

    def __init__(self, name: str) -> None:
        self._commands: list[Command] = []
        self._parent_to_palette_map: dict[int, QCommandPalette] = {}
        self._palette_to_parent_map: _WVDict = weakref.WeakValueDictionary()
        self._name = name

    @property
    def commands(self) -> list[Command]:
        """List of all the commands."""
        return self._commands.copy()

    def register(self, cmd: CommandRule) -> None:
        """Register a command to the palette."""
        # update defaults
        sep = ':' if ':' in cmd.id else '.'
        *contexts, _ = cmd.id.split(sep)
        title = ' > '.join(contexts)
        enablement = cmd.enablement or Constant(True)
        desc = cmd.title
        tooltip = cmd.status_tip or ''
        cmd = Command(cmd, title, desc, tooltip, enablement)
        self._commands.append(cmd)
        return

    def get_widget(self, parent: QtW.QWidget) -> QCommandPalette:
        """Get a command palette widget for the given parent widget."""
        from napari._qt.widgets.qt_command_palette import QCommandPalette

        _id = id(parent)
        if (widget := self._parent_to_palette_map.get(_id)) is None:
            widget = QCommandPalette(parent)
            widget.extend_command(self._commands)
            self._parent_to_palette_map[_id] = widget
            self._palette_to_parent_map[id(widget)] = parent
        return widget


_GLOBAL_PALETTES: dict[str, CommandPalette] = {}


def get_palette(name) -> CommandPalette:
    """
    Get the global command palette object.

    Examples
    --------
    >>> palette = get_palette()  # get the default palette
    >>> palette = get_palette("my_module")  # get a palette for specific app
    """
    global _GLOBAL_PALETTES

    if not isinstance(name, str):
        raise TypeError(f'Expected str, got {type(name).__name__}')
    if (palette := _GLOBAL_PALETTES.get(name)) is None:
        palette = _GLOBAL_PALETTES[name] = CommandPalette(name=name)
    return palette
