from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Callable, TypeVar, overload

from app_model.expressions import Constant

from napari.components.command_palette._components import Command

if TYPE_CHECKING:
    from app_model.types import CommandRule
    from qtpy import QtWidgets as QtW

    from napari._qt.qt_main_window import _QtMainWindow
    from napari._qt.widgets.qt_command_palette import QCommandPalette

    _WVDict = weakref.WeakValueDictionary[int, _QtMainWindow]

_F = TypeVar('_F', bound=Callable)


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

    def add_group(self, title: str) -> CommandGroup:
        """Add a group to the command palette."""
        return CommandGroup(title, parent=self)

    def get_widget(self, parent: _QtMainWindow) -> QCommandPalette:
        """Get a command palette widget for the given parent widget."""
        from napari._qt.widgets.qt_command_palette import QCommandPalette

        _id = id(parent)
        if (widget := self._parent_to_palette_map.get(_id)) is None:
            widget = QCommandPalette(parent)
            widget.extend_command(self._commands)
            self._parent_to_palette_map[_id] = widget
            self._palette_to_parent_map[id(widget)] = parent
        return widget

    def show_widget(self, parent: _QtMainWindow) -> None:
        """Show command palette widget."""
        # update all the context
        return self.get_widget(parent).show()

    def install(self, parent: QtW.QWidget) -> QCommandPalette:
        """
        Install command palette on a Qt widget.

        Parameters
        ----------
        parent : QtW.QWidget
            The widget to install on.
        keys : str, optional
            If given, this key sequence will be used to show the command palette.
        """
        widget = self.get_widget(parent)
        widget.install_to(parent)
        return widget

    def update(self, parent: QtW.QWidget | None = None):
        """Update command palette install to the given parent widget."""
        if parent is None:
            for p in self._palette_to_parent_map.values():
                self.update(p)
            return
        _id = id(parent)
        widget = self._parent_to_palette_map[_id]
        widget.clear_commands()
        widget.extend_command(self._commands)
        return


class CommandGroup:
    def __init__(self, title: str, parent: CommandPalette) -> None:
        self._palette_ref = weakref.ref(parent)
        self._title = title

    def __repr__(self) -> str:
        return f'CommandGroup<{self.title}>'

    @property
    def palette(self) -> CommandPalette:
        """The parent command palette object."""
        if palette := self._palette_ref():
            return palette
        raise RuntimeError('CommandPalette is already destroyed.')

    @property
    def title(self) -> str:
        """The title of this group."""
        return self._title

    @overload
    def register(
        self,
        func: _F,
        desc: str | None = None,
        tooltip: str | None = None,
        enablement: Callable[..., bool] | None = None,
    ) -> _F: ...

    @overload
    def register(
        self,
        desc: str | None = None,
        tooltip: str | None = None,
        enablement: Callable[..., bool] | None = None,
    ) -> Callable[[_F], _F]: ...

    def register(self, *args, **kwargs):
        if 'title' in kwargs:
            raise TypeError(
                "register() got an unexpected keyword argument 'title'"
            )
        if len(args) > 0:
            if callable(args[0]):
                args = args[:1] + (self.title,) + args[1:]
            else:
                args = (self.title,) + args
        else:
            args = (self.title,)

        return self.palette.register(*args, **kwargs)


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
