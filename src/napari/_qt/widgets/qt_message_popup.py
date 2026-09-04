from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout, QWidget

from napari._qt.qt_resources import get_stylesheet
from napari.settings import get_settings


class WarnPopup(QDialog):
    """Dialog to inform user that shortcut is already assigned."""

    def __init__(
        self,
        parent: QWidget | None = None,
        text: str = '',
    ) -> None:
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
        self._xbutton.setStyleSheet('background-color: rgba(0, 0, 0, 0);')

        # Layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self._message)

        self.setLayout(main_layout)

        settings = get_settings()
        font_size = settings.appearance.font_size
        extra_variables = {'font_size': f'{font_size}pt'}

        self.setStyleSheet(
            get_stylesheet(
                settings.appearance.theme, extra_variables=extra_variables
            )
        )
        self._xbutton.raise_()

    def _close(self) -> None:
        self.close()
