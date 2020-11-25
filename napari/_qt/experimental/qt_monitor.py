"""QtMonitor
"""
from qtpy.QtCore import QObject, QTimer

from ...components.camera import Camera
from ...components.experimental.monitor import monitor

POLL_INTERVAL_MS = 33  # About 30Hz.


class QtMonitor(QObject):
    """Polls the monitor service.

    The goal is really to poll the service "once per frame" but we tie
    into the camera as a proxy for that for now. We also might need to
    poll based on a timer, one reason why this is a QObject.

    Parameters
    ----------
    parent : QObject
        Parent Qt object.
    """

    def __init__(self, parent: QObject, camera: Camera):
        super().__init__(parent)
        camera.events.center.connect(self._on_camera_move)

        # For sending messages polling during _on_camera_move() might be
        # enough. But for receiving messages right now we need to poll all
        # the time. Although we can probably get rid of that requirement
        # when we are bit fancier inside the MonitorServer as far as
        # receiving data.
        self.timer = QTimer()
        self.timer.setInterval(POLL_INTERVAL_MS)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start()

    def _on_camera_move(self, _event) -> None:
        """Called when camera was moved."""
        monitor.poll()

    def _on_timer(self) -> None:
        """Called every POLL_INTERVAL_MS milliseconds."""
        monitor.poll()
