from __future__ import annotations

from threading import Event
from typing import TYPE_CHECKING
from weakref import ref

from qtpy.QtCore import QObject, QThread, Signal

from napari.utils.notifications import Notification, notification_manager

if TYPE_CHECKING:
    from napari.components import ViewerModel


class StatusChecker(QThread):
    status_and_tooltip_changed = Signal(object)

    def __init__(self, viewer: ViewerModel, parent: QObject | None = None):
        super().__init__(parent=parent)
        self.viewer_ref = ref(viewer)
        self._event = Event()
        self._need_status_update = False
        self._terminate = False

    def trigger_status_update(self) -> None:
        self._need_status_update = True
        self._event.set()

    def terminate(self) -> None:
        self._terminate = True
        self._event.set()

    def run(self) -> None:
        self._event.clear()
        while not self._terminate:
            if self._need_status_update:
                self._need_status_update = False
                self._event.clear()
                self.calculate_status()
            else:
                self._event.wait()

    def calculate_status(self) -> None:
        viewer = self.viewer_ref()
        if viewer is None:
            return

        try:
            self.status_and_tooltip_changed.emit(
                viewer._calc_status_from_cursor()
            )
        except Exception as e:  # pragma: no cover # noqa: BLE001
            notification_manager.dispatch(Notification.from_exception(e))
