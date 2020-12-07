"""QtPoll class.

Poll visuals or other objects so they can do things even when the
mouse/camera are not moving. Usually for just a short period of time.
"""
from qtpy.QtCore import QEvent, QObject, QTimer

from ...components.camera import Camera
from ...utils.events import EmitterGroup

POLL_INTERVAL_MS = 16.666  # About 60HZ


class QtPoll(QObject):
    """Polls anything once per frame via an event.

    QtPoll was first created for VispyTiledImageLayer. It polls the visual
    when the camera moves. However, we also want visuals to keep loading
    chunks even when the camera stops. We want the visual to finish up
    anything that was in progress. Before it goes fully idle.

    QtPoll will poll those visuals using a timer. If the visual says the
    event was "handled" it means the visual has more work to do. If that
    happens, QtPoll will continue to poll and draw the visual it until the
    visual is done with the in-progress work.

    An analogy is a snow globe. The user moving the camera shakes up the
    snow globe. We need to keep polling/drawing things until all the snow
    settles down. Then everything will stay completely still until the
    camera is moved again, shaking up the globe once more.

    Parameters
    ----------
    parent : QObject
        Parent Qt object.
    camera : Camera
        The viewer's main camera.
    """

    def __init__(self, parent: QObject, camera: Camera):
        super().__init__(parent)

        self.events = EmitterGroup(source=self, auto_connect=True, poll=None)
        camera.events.connect(self._on_camera)

        self.timer = QTimer()
        self.timer.setInterval(POLL_INTERVAL_MS)
        self.timer.timeout.connect(self._on_timer)

    def _on_camera(self, _event) -> None:
        """Called when camera view changes at all."""
        # Poll right away. If the timer is running, it's generally starved
        # out by the mouse button being down. Not sure why. If we end up
        # "double polling" it *should* be harmless. However, if we don't
        # poll then everything is frozen. So better to poll.
        self._poll()

        # Start the timer so that we will keep polling even if the camera
        # doesn't move again. Although the mouse movement is starving out
        # the timer right now, we need the timer going so we keep polling
        # even if the mouse stops.
        self.timer.start()

    def _on_timer(self) -> None:
        """Called when the timer is running."""
        # The timer is running which means someone we are polling still has
        # work to do.
        self._poll()

    def _poll(self) -> None:
        """Poll everyone listening to our event."""
        event = self.events.poll()

        if not event.handled:
            # No one needed polling, so stop the timer. No polling will
            # happen until the camera moves again.
            self.timer.stop()
            return

        # Someone handled the event. They need to be polled even if the
        # camera stops moving. So start the timer. They can finish their
        # work and continue to be drawn even while the camera is stopped.
        self.timer.start()

    def closeEvent(self, _event: QEvent) -> None:
        """Cleanup and close.

        Parameters
        ----------
        event : QEvent
            The close event.
        """
        self.timer.stop()
        self.deleteLater()
