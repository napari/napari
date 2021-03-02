from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QSplashScreen

from ..qt_event_loop import NAPARI_ICON_PATH, get_app


class NapariSplashScreen(QSplashScreen):
    def __init__(self, width=360):
        get_app()
        pm = QPixmap(NAPARI_ICON_PATH).scaled(
            width, width, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        super().__init__(pm)
        self.show()
