"""QtMonitor
"""
from qtpy.QtCore import QObject, QTimer

from ...components.camera import Camera
from ...components.experimental.monitor import monitor

POLL_INTERVAL_MS = 33  # About 30Hz.


class QtMonitor(QObject):
    """Polls the monitor service.

    This object is only created if NAPARI_MON is set.

    To send data, it's only necessary to poll when the camera moves.
    However to receive commands right now we need to poll all the time,
    with a timer. We should be able to rid of timed polling eventually.

    Parameters
    ----------
    parent : QObject
        Parent Qt object.
    """

    def __init__(self, parent: QObject, camera: Camera):
        super().__init__(parent)
        camera.events.connect(self._on_camera_move)

        self.timer = QTimer()
        self.timer.setInterval(POLL_INTERVAL_MS)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start()

    @staticmethod
    def _on_camera_move(_event) -> None:
        """Called when camera was moved."""
        monitor.poll()

    @staticmethod
    def _on_timer() -> None:
        """Called every POLL_INTERVAL_MS milliseconds."""
        monitor.poll()
