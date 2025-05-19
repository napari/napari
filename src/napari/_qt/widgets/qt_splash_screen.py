from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QSplashScreen

from napari._qt.qt_event_loop import NAPARI_ICON_PATH, get_qapp


class NapariSplashScreen(QSplashScreen):
    def __init__(self, width=360) -> None:
        get_qapp()
        pm = QPixmap(NAPARI_ICON_PATH).scaled(
            width,
            width,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        super().__init__(pm)
        self.show()
