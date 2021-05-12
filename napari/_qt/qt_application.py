from qtpy.QtCore import Slot
from qtpy.QtWidgets import QApplication

from .dialogs.qt_notification import NapariQtNotification


class NapariQApplication(QApplication):
    """
    Extend QApplication to handle notfications from `threading.threads`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Use to handle notifications raised by non managed threads
        # See: napari._qt.qt_eventloop._show_notifications
        self._notification = None

    @Slot()
    def show_notification(self):
        if self._notification:
            NapariQtNotification.show_notification(self._notification)
            self._notification = None
