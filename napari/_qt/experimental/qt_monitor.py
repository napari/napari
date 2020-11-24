"""QtMonitor
"""
from qtpy.QtCore import QObject

from ...components.camera import Camera
from ...components.experimental.monitor import monitor


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

    def _on_camera_move(self, _event):
        monitor.poll()
