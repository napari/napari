from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...utils.translations import trans


class ConfirmDialog(QDialog):
    """Dialog to confirms a user's choice to restore default settings."""

    valueChanged = Signal(bool)

    def __init__(
        self,
        parent: QWidget = None,
        text: str = "",
    ):
        super().__init__(parent)

        # Set up components
        self._question = QLabel(self)
        self._button_restore = QPushButton(trans._("Restore"))
        self._button_cancel = QPushButton(trans._("Cancel"))
        self._button_restore.setDefault(True)

        # Widget set up
        self._question.setText(text)

        # Layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self._button_cancel)
        button_layout.addWidget(self._button_restore)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self._question)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

        # Signals
        self._button_cancel.clicked.connect(self.on_click_cancel)
        self._button_restore.clicked.connect(self.on_click_restore)

    def on_click_cancel(self):
        """Do not restore defaults and close window."""
        self.valueChanged.emit(False)
        self.close()

    def on_click_restore(self):
        """Emit signal to restore defaults and close window."""
        self.valueChanged.emit(True)
        self.close()


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
