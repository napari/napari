"""Debug menu.

The debug menu is for developer-focused functionality that we want to be
easy-to-use and discoverable, but which is not for the average user.

Current Items
-------------
Trace File -> Start Tracing...
Trace File -> Stop Tracking
"""
import os

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QAction, QFileDialog

from ..utils import perf


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
        self.debug_menu = main_window.main_menu.addMenu('&Debug')

        self.perf = PerformanceSubMenu(
            main_window, self.debug_menu.addMenu("Performance Trace")
        )


class PerformanceSubMenu:
    """The flyout menu to start/stop recording a trace file.
    """

    def __init__(self, main_window, sub_menu):
        self.main_window = main_window
        self.sub_menu = sub_menu
        self.start = self._add_start()
        self.stop = self._add_stop()
        self._set_recording(False)

        # If NAPARI_TRACE_FILE is set we immediately start tracing. This is
        # easier than manually starting the trace from the debug menu in
        # some cases.
        path = os.getenv("NAPARI_TRACE_FILE")
        if path is not None:
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
        """Add Start Recording action.
        """
        start = QAction('Start Recording...', self.main_window._qt_window)
        start.setShortcut('Alt+T')
        start.setStatusTip('Start recording a trace file')
        start.triggered.connect(self._start_trace_dialog)
        self.sub_menu.addAction(start)
        return start

    def _add_stop(self):
        """Add Stop Recording action.
        """
        stop = QAction('Stop Recording', self.main_window._qt_window)
        stop.setShortcut('Shift+Alt+T')
        stop.setStatusTip('Stop recording a trace file')
        stop.triggered.connect(self._stop_trace)
        self.sub_menu.addAction(stop)
        return stop

    def _start_trace_dialog(self):
        """Open Save As dialog to start recording a trace file."""
        viewer = self.main_window.qt_viewer

        filename, _ = QFileDialog.getSaveFileName(
            parent=viewer,
            caption='Record performance trace file',
            directory=viewer._last_visited_dir,
            filter="Trace Files (*.json)",
        )
        if filename:
            filename = _ensure_extension(filename, '.json')

            def start_trace():
                self._start_trace(filename)

            # Schedule this to avoid bogus "MetaCall" event for the entire
            # time the file dialog was up.
            QTimer.singleShot(0, start_trace)

    def _start_trace(self, path: str):
        perf.timers.start_trace_file(path)
        self._set_recording(True)

    def _stop_trace(self):
        """Stop recording a trace file.
        """
        perf.timers.stop_trace_file()
        self._set_recording(False)
