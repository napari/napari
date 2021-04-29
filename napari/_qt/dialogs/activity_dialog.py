from PyQt5.QtWidgets import QWidget
from qtpy.QtCore import QSize, Signal
from qtpy.QtWidgets import QDialog, QScrollArea, QSizePolicy, QVBoxLayout

MIN_WIDTH = 400
MIN_HEIGHT = 120


class ActivityDialog(QDialog):
    """Activity Dialog for Napari progress bars."""

    resized = Signal(QSize)
    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('Activity')
        self.setMinimumWidth(MIN_WIDTH)
        self.setMinimumHeight(MIN_HEIGHT)
        self.setMaximumHeight(MIN_HEIGHT)
        self.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )

        self.baseWidget = QWidget()

        self.activity_layout = QVBoxLayout()
        self.activity_layout.addStretch()
        self.baseWidget.setLayout(self.activity_layout)

        self.scroll = QScrollArea()
        self.scroll.setWidget(self.baseWidget)
