from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout

from napari._qt.qt_resources import get_stylesheet
from napari.settings import get_settings


class WarnPopup(QDialog):
    """Dialog to inform user that shortcut is already assigned."""

    def __init__(
        self,
        parent=None,
        text: str = "",
    ):
        super().__init__(parent)

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

        # Widgets
        self._message = QLabel()
        self._xbutton = QPushButton('x', self)
        self._xbutton.setFixedSize(20, 20)

        # Widget set up
        self._message.setText(text)
        self._message.setWordWrap(True)
        self._xbutton.clicked.connect(self._close)
        self._xbutton.setStyleSheet("background-color: rgba(0, 0, 0, 0);")

        # Layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self._message)

        self.setLayout(main_layout)

        self.setStyleSheet(get_stylesheet(get_settings().appearance.theme))
        self._xbutton.raise_()

    def _close(self):
        self.close()
