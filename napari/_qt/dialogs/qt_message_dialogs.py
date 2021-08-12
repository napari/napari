from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QDialog,
    QGridLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...utils.translations import trans


class ResetNapariInfoDialog(QDialog):
    """Dialog to inform the user that restart of Napari is necessary to enable setting."""

    valueChanged = Signal()

    def __init__(
        self,
        parent: QWidget = None,
        text: str = "",
    ):
        super().__init__(parent)
        # Set up components
        self._info_str = QLabel(self)
        self._button_ok = QPushButton(trans._("OK"))
        # Widget set up
        self._info_str.setText(text)

        # Layout
        button_layout = QGridLayout()
        button_layout.addWidget(self._button_ok, 0, 1)
        button_layout.setColumnStretch(0, 1)
        button_layout.setColumnStretch(1, 1)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self._info_str)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

        # Signals
        self._button_ok.clicked.connect(self._close_dialog)

    def _close_dialog(self):
        """Close window."""
        self.close()
