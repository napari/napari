"""QtPoll class.
"""
from qtpy.QtCore import QObject, QTimer

from ...components.camera import Camera
from ...utils.events import EmitterGroup

POLL_INTERVAL_MS = 16.666  # About 60HZ


class QtPoll(QObject):
    """Polls anything once per frame via an event.

    Created for VispyTiledImageLayer. We poll the visual when the camera
    moves. But the visuals sometimes load multiple chunks amortized over a
    number of frames. And it needs to continue to do that, continue
    loading chunks, even though the mouse is not moving.

    QtPoll will poll those visuals using a timer, until they report they
    are done and no longer need polling. Then we go quiet, and nothing will
    be polled until the camera moves again.

    An analogy is a snow globe. The user moving the mouse is shaking up
    the snow globe. And we need to keep polling/updating things until
    all the flakes settle down. Then everything will stay 100% still
    until the mouse is moved again.

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
        if self.timer.isActive():
            # Do nothing since the timer is already going. It's possible we
            # should poll anyway, and risk double poll? But for now let's
            # let the timer do it until we know more.
            return

        # Poll right away to kick things off instantly.
        self._poll()

        # Start the timer so that we keep polling even if the camera
        # doesn't move again.
        self.timer.start()

    def _on_timer(self) -> None:
        """Called when the timer is running."""
        self._poll()

    def _poll(self) -> None:
        """Poll everyone listening to our event."""
        event = self.events.poll()

        if not event.handled:
            # No one needed polling, so stop the timer, no polling will
            # happen until the camera moves again.
            self.timer.stop()
            return

        # Somone handled the event. They need to be polled even if the
        # camera stops moving. So start the timer. So they can finish
        # loading data or animating something.
        self.timer.start()
