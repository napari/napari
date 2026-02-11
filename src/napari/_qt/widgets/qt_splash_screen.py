from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QSplashScreen

from napari._qt.qt_event_loop import get_icon_path, get_qapp


class NapariSplashScreen(QSplashScreen):
    def __init__(self, width=360) -> None:
        get_qapp()
        pm = QPixmap(str(get_icon_path())).scaled(
            width,
            width,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        super().__init__(pm)
        self.show()
