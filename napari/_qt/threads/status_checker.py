"""A performant, dedicated thread to compute cursor status and signal updates to a viewer."""

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
    """A dedicated thread for performant updating of the status bar.

    This class offloads the job of computing the cursor status into a separate thread,
    Qt Signals are used to update the main viewer with the status string.

    Prior to https://github.com/napari/napari/pull/7146, status bar updates
    happened on the main thread in the viewer model, which could be
    expensive in some layers and resulted in bad performance when some
    layers were selected.

    Because the thread runs a single infinite while loop, the updates are
    naturally throttled since they can only be sent at the rate which updates
    can be computed, but no faster.

    Attributes
    ----------
    _need_status_update : threading.Event
        An Event (fancy thread-safe bool-like to synchronize threads)
        for keeping track of when the status needs updating
        (because the cursor has moved).
    _terminate : bool
        If set to True, the status checker thread needs to be terminated.
        When the QtViewer is being closed, it sets this flag to terminate
        the status checker thread.
        After _terminate is set to True, no more status updates are sent.
        Default: False.
    viewer_ref : weakref.ref[napari.viewer.ViewerModel]
        A weak reference to the viewer which is providing status updates.
        We keep a weak reference to the viewer so the status checker thread
        will not prevent the viewer from being garbage collected.
        We proactively check the viewer to determine if a new status update
        needs to be computed and emitted.
    """

    # Create a Signal to establish a lightweight communication mechanism between the
    # viewer and the status checker thread for cursor events and related status
    status_and_tooltip_changed = Signal(object)

    def __init__(self, viewer: ViewerModel, parent: QObject | None = None):
        super().__init__(parent=parent)
        self.viewer_ref = ref(viewer)
        self._need_status_update = Event()
        self._need_status_update.clear()
        self._terminate = False

    def trigger_status_update(self) -> None:
        """Trigger a status update computation.

        When the cursor moves, the viewer will call this to instruct
        the status checker to update the viewer with the present status.
        """
        self._need_status_update.set()

    def terminate(self) -> None:
        """Terminate the status checker thread.

        For proper cleanup,it's important to set _terminate to True before
        calling _needs_status_update.set.
        """
        self._terminate = True
        self._need_status_update.set()

    def start(
        self, priority: QThread.Priority = QThread.Priority.InheritPriority
    ) -> None:
        """Start the status checker thread.

        Make sure to set the _terminate attribute to False prior to start.
        """
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

        If the viewer is not available, do nothing. Otherwise,
        emit the signal that the status has changed.
        """
        viewer = self.viewer_ref()
        if viewer is None:
            return

        try:
            # Calculate the status change from cursor's movement
            res = viewer._calc_status_from_cursor()
        except Exception as e:  # pragma: no cover # noqa: BLE001
            # Our codebase is not threadsafe. It is possible that an
            # ViewerModel or Layer state is changed while we are trying to
            # calculate the status, which may cause an exception.
            # All exceptions are caught and handled to keep updates
            # from crashing the thread. The exception is logged
            # and a notification is sent.
            notification_manager.dispatch(Notification.from_exception(e))
            return
        # Emit the signal with the updated status
        self.status_and_tooltip_changed.emit(res)


if os.environ.get('ASV') == 'true':
    # This is a hack to make sure that the StatusChecker thread is not
    # running when the benchmark is running. This is because the
    # StatusChecker thread may introduce some noise in the benchmark
    # results from waiting on its termination.
    StatusChecker.start = lambda self, priority=0: None  # type: ignore[assignment]
