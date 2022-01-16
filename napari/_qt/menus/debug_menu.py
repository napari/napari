"""Debug menu.

The debug menu is for developer-focused functionality that we want to be
easy-to-use and discoverable, but which is not for the average user.

"""
from typing import TYPE_CHECKING

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QFileDialog

from ...utils import perf
from ...utils.history import get_save_history, update_save_history
from ...utils.translations import trans
from ._util import NapariMenu, populate_menu

if TYPE_CHECKING:
    from ..qt_main_window import Window


class DebugMenu(NapariMenu):
    def __init__(self, window: 'Window'):
        self._win = window
        super().__init__(trans._('&Debug'), window._qt_window)
        self._perf_menu = NapariMenu(trans._("Performance Trace"), self)

        ACTIONS = [
            {
                'menu': self._perf_menu,
                'items': [
                    {
                        'text': trans._('Start Recording...'),
                        'slot': self._start_trace_dialog,
                        'shortcut': 'Alt+T',
                        'statusTip': trans._('Start recording a trace file'),
                    },
                    {
                        'text': trans._('Stop Recording...'),
                        'slot': self._stop_trace,
                        'shortcut': 'Shift+Alt+T',
                        'statusTip': trans._('Stop recording a trace file'),
                    },
                ],
            }
        ]
        populate_menu(self, ACTIONS)
        self._set_recording(False)
        if perf.perf_config:
            path = perf.perf_config.trace_file_on_start
            if path is not None:
                # Config option "trace_file_on_start" means immediately
                # start tracing to that file. This is very useful if you
                # want to create a trace every time you start napari,
                # without having to start it from the debug menu.
                self._start_trace(path)

    def _start_trace_dialog(self):
        """Open Save As dialog to start recording a trace file."""
        viewer = self._win._qt_viewer

        dlg = QFileDialog()
        hist = get_save_history()
        dlg.setHistory(hist)
        filename, _ = dlg.getSaveFileName(
            parent=viewer,
            caption=trans._('Record performance trace file'),
            directory=hist[0],
            filter=trans._("Trace Files (*.json)"),
        )
        if filename:
            if not filename.endswith(".json"):
                filename += ".json"

            # Schedule this to avoid bogus "MetaCall" event for the entire
            # time the file dialog was up.
            QTimer.singleShot(0, lambda: self._start_trace(filename))

            update_save_history(filename)

    def _start_trace(self, path: str):
        perf.timers.start_trace_file(path)
        self._set_recording(True)

    def _stop_trace(self):
        """Stop recording a trace file."""
        perf.timers.stop_trace_file()
        self._set_recording(False)

    def _set_recording(self, recording: bool):
        """Toggle which are enabled/disabled.

        Parameters
        ----------
        recording : bool
            Are we currently recording a trace file.
        """
        for action in self._perf_menu.actions():
            if trans._('Start Recording') in action.text():
                action.setEnabled(not recording)
            elif trans._('Stop Recording') in action.text():
                action.setEnabled(recording)
