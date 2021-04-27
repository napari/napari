"""Debug menu.

The debug menu is for developer-focused functionality that we want to be
easy-to-use and discoverable, but which is not for the average user.

Menu Items
----------
Trace File -> Start Tracing...
Trace File -> Stop Tracing
"""
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QAction, QFileDialog

from ...utils import perf
from ...utils.history import get_save_history, update_save_history
from ...utils.translations import trans


def _ensure_extension(filename: str, extension: str):
    """Add the extension if needed."""
    if filename.endswith(extension):
        return filename
    return filename + extension


class DebugMenu:
    def __init__(self, main_window):
        """Create the debug menu.

        Parameters
        ----------
        main_window : qtpy.QtWidgets.QMainWindow.menuBar
            We add ourselves to this menu.
        """
        self.debug_menu = main_window.main_menu.addMenu(trans._('&Debug'))

        self.perf = PerformanceSubMenu(
            main_window, self.debug_menu.addMenu(trans._("Performance Trace"))
        )


class PerformanceSubMenu:
    """The flyout menu to start/stop recording a trace file."""

    def __init__(self, main_window, sub_menu):
        self.main_window = main_window
        self.sub_menu = sub_menu
        self.start = self._add_start()
        self.stop = self._add_stop()
        self._set_recording(False)

        if perf.perf_config:
            path = perf.perf_config.trace_file_on_start
            if path is not None:
                # Config option "trace_file_on_start" means immediately
                # start tracing to that file. This is very useful if you
                # want to create a trace every time you start napari,
                # without having to start it from the debug menu.
                self._start_trace(path)

    def _set_recording(self, recording: bool):
        """Toggle which are enabled/disabled.

        Parameters
        ----------
        recording : bool
            Are we currently recording a trace file.
        """
        self.start.setEnabled(not recording)
        self.stop.setEnabled(recording)

    def _add_start(self):
        """Add Start Recording action."""
        start = QAction(
            trans._('Start Recording...'), self.main_window._qt_window
        )
        start.setShortcut('Alt+T')
        start.setStatusTip(trans._('Start recording a trace file'))
        start.triggered.connect(self._start_trace_dialog)
        self.sub_menu.addAction(start)
        return start

    def _add_stop(self):
        """Add Stop Recording action."""
        stop = QAction(trans._('Stop Recording'), self.main_window._qt_window)
        stop.setShortcut('Shift+Alt+T')
        stop.setStatusTip(trans._('Stop recording a trace file'))
        stop.triggered.connect(self._stop_trace)
        self.sub_menu.addAction(stop)
        return stop

    def _start_trace_dialog(self):
        """Open Save As dialog to start recording a trace file."""
        viewer = self.main_window.qt_viewer

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
            filename = _ensure_extension(filename, '.json')

            def start_trace():
                self._start_trace(filename)

            # Schedule this to avoid bogus "MetaCall" event for the entire
            # time the file dialog was up.
            QTimer.singleShot(0, start_trace)

            update_save_history(filename)

    def _start_trace(self, path: str):
        perf.timers.start_trace_file(path)
        self._set_recording(True)

    def _stop_trace(self):
        """Stop recording a trace file."""
        perf.timers.stop_trace_file()
        self._set_recording(False)
