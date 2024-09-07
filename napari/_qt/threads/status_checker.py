from __future__ import annotations

import os
from threading import Event
from typing import TYPE_CHECKING
from weakref import ref

from qtpy.QtCore import QObject, QThread, Signal

from napari.utils.notifications import Notification, notification_manager

if TYPE_CHECKING:
    from napari.components import ViewerModel


class StatusChecker(QThread):
    """Separate thread dedicated to updating the status bar.

    Prior to https://github.com/napari/napari/pull/7146, status bar updates
    happened on the main thread in the viewer model, which could be
    expensive in some layers and resulted in bad performance when some
    layers were selected.

    This class puts the job of computing the status into a separate thread,
    which then uses Qt Signals to update the main viewer with the status
    string.

    Because the thread runs a single infinite while loop, the updates are
    naturally throttled to as fast as they can be computed, but no faster.

    Attributes
    ----------
    _need_status_update : threading.Event
        An Event (fancy thread-safe bool-like to synchronize threads)
        for keeping track of when the status needs updating
        (because the cursor has moved).
    _terminate : bool
        Whether the thread needs to be terminated. Set by the QtViewer
        when it is being closed. No more status updates will take place if
        _terminate is set.
    viewer_ref : weakref.ref[napari.viewer.ViewerModel]
        A weak reference to the viewer providing the status updates. We
        don't want this thread to prevent the viewer from being garbage
        collected, so we keep only a weak reference, and check it when
        we need to compute a new status update.
    """

    status_and_tooltip_changed = Signal(object)

    def __init__(self, viewer: ViewerModel, parent: QObject | None = None):
        super().__init__(parent=parent)
        self.viewer_ref = ref(viewer)
        self._need_status_update = Event()
        self._need_status_update.clear()
        self._terminate = False

    def trigger_status_update(self) -> None:
        """Trigger a status update computation.

        The viewer should call this when the cursor moves.
        """
        self._need_status_update.set()

    def terminate(self) -> None:
        """Terminate the thread.

        It's important that _terminate is set to True before
        _needs_status_update.set is called.
        """
        self._terminate = True
        self._need_status_update.set()

    def start(
        self, priority: QThread.Priority = QThread.Priority.InheritPriority
    ) -> None:
        self._terminate = False
        super().start(priority)

    def run(self) -> None:
        while not self._terminate:
            if self.viewer_ref() is None:
                # Stop thread when viewer is closed
                return
            if self._need_status_update.is_set():
                self._need_status_update.clear()
                self.calculate_status()
            else:
                self._need_status_update.wait()

    def calculate_status(self) -> None:
        """Calculate the status and emit the signal.

        If the viewer is not available, do nothing.
        """
        viewer = self.viewer_ref()
        if viewer is None:
            return

        try:
            res = viewer._calc_status_from_cursor()
        except Exception as e:  # pragma: no cover # noqa: BLE001
            # Our codebase is not threadsafe. It is possible that an
            # ViewerModel or Layer state is changed while we are trying to
            # calculate the status, which may lead to an exception.
            # We catch all exceptions here to prevent the thread from
            # crashing. The error is logged and a notification is shown.
            #
            # We do not want to crash the thread to keep the status updates.
            notification_manager.dispatch(Notification.from_exception(e))
        self.status_and_tooltip_changed.emit(res)


if os.environ.get('ASV') == 'true':
    # This is a hack to make sure that the StatusChecker thread is not
    # running when the benchmark is running. This is because the
    # StatusChecker thread may introduce some noise in the benchmark
    # results from waiting on its termination.
    StatusChecker.start = lambda self, priority=0: None  # type: ignore[assignment]
