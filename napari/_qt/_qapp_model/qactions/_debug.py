"""Qt 'Debug' menu actions.

The debug menu is for developer-focused functionality that we want to be
easy-to-use and discoverable, but is not for the average user.
"""

from app_model.types import Action, KeyCode, KeyMod, SubmenuItem
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QFileDialog

from napari._app_model.constants import MenuGroup, MenuId
from napari._qt.qt_viewer import QtViewer
from napari.utils import perf
from napari.utils.history import get_save_history, update_save_history
from napari.utils.translations import trans

# Debug submenu
DEBUG_SUBMENUS = [
    (
        MenuId.MENUBAR_DEBUG,
        SubmenuItem(
            submenu=MenuId.DEBUG_PERFORMANCE,
            title=trans._('Performance Trace'),
        ),
    ),
]


def _start_trace_dialog(qt_viewer: QtViewer) -> None:
    """Open Save As dialog to start recording a trace file."""
    dlg = QFileDialog()
    hist = get_save_history()
    dlg.setHistory(hist)
    filename, _ = dlg.getSaveFileName(
        qt_viewer,  # parent
        trans._('Record performance trace file'),  # caption
        hist[0],  # directory in PyQt, dir in PySide
        filter=trans._('Trace Files (*.json)'),
    )
    if filename:
        if not filename.endswith('.json'):
            filename += '.json'

        # Schedule this to avoid bogus "MetaCall" event for the entire
        # time the file dialog was up.
        QTimer.singleShot(0, lambda: _start_trace(filename))

        update_save_history(filename)


def _start_trace(path: str) -> None:
    """Start recording a trace file."""
    perf.timers.start_trace_file(path)


def _stop_trace() -> None:
    """Stop recording a trace file."""
    perf.timers.stop_trace_file()


def _is_set_trace_active() -> bool:
    """Whether we are currently recording a set trace."""
    return perf.timers.trace_file is not None


Q_DEBUG_ACTIONS: list[Action] = [
    Action(
        id='napari.window.debug.start_trace_dialog',
        title=trans._('Start Recording...'),
        callback=_start_trace_dialog,
        menus=[
            {'id': MenuId.DEBUG_PERFORMANCE, 'group': MenuGroup.NAVIGATION}
        ],
        keybindings=[{'primary': KeyMod.Alt | KeyCode.KeyT}],
        enablement='not is_set_trace_active',
        status_tip=trans._('Start recording a trace file'),
    ),
    Action(
        id='napari.window.debug.stop_trace',
        title=trans._('Stop Recording...'),
        callback=_stop_trace,
        menus=[
            {'id': MenuId.DEBUG_PERFORMANCE, 'group': MenuGroup.NAVIGATION}
        ],
        keybindings=[{'primary': KeyMod.Alt | KeyMod.Shift | KeyCode.KeyT}],
        enablement='is_set_trace_active',
        status_tip=trans._('Stop recording a trace file'),
    ),
]
