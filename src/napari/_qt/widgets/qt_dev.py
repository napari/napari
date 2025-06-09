"""Development widgets."""

import logging
import sys
import typing as ty
from contextlib import suppress

logger = logging.getLogger()

if ty.TYPE_CHECKING:
    from qtreload.qt_reload import QtReloadWidget


DEFAULT_HOOK = None


def debugger_hook(error_type, value, tb) -> None:
    """Drop into python debugger on uncaught exception."""
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode, or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(error_type, value, tb)
    else:
        import pdb
        import traceback

        # we are NOT in interactive mode, print the exception...
        with suppress(Exception):
            traceback.print_exception(error_type, value, tb)
            pdb.post_mortem(tb)


def install_debugger_hook() -> None:
    """Activate the debugger hook."""
    global DEFAULT_HOOK

    if DEFAULT_HOOK is None:
        DEFAULT_HOOK = sys.excepthook
    sys.excepthook = debugger_hook


def qdev(
    parent=None,
    modules: ty.Iterable[str] = ('napari', 'napari_builtins'),
) -> 'QtReloadWidget':
    """Create reload widget."""
    from qtreload.qt_reload import QtReloadWidget

    logger.debug('Creating reload widget for modules: {}.', modules)
    return QtReloadWidget(modules, parent=parent)
