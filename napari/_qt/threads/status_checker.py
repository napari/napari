from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING
from weakref import ref

from qtpy.QtCore import QObject, QThread, Signal

if TYPE_CHECKING:
    from napari.components import ViewerModel


class StatusChecker(QThread):
    status_and_tooltip_changed = Signal(object)

    def __init__(self, viewer: ViewerModel, parent: QObject | None = None):
        super().__init__(parent=parent)
        self.viewer_ref = ref(viewer)
        self._lock = Lock()
        self._need_status_update = False

    def trigger_status_update(self):
        self._need_status_update = True
        if self._lock.locked():
            self._lock.release()

    def run(self):
        self._lock.acquire()
        while True:
            if self._need_status_update:
                self._need_status_update = False
                self.calculate_status()
            else:
                self._lock.acquire()

    def calculate_status(self):
        viewer = self.viewer_ref()
        if viewer is None:
            return

        self.status_and_tooltip_changed.emit(viewer._calc_status_from_cursor())
