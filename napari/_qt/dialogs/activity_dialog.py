from qtpy.QtCore import QSize, Signal
from qtpy.QtWidgets import QDialog, QVBoxLayout

from ...utils.translations import trans


class ActivityDialog(QDialog):
    """Activity Dialog for Napari progress bars."""

    resized = Signal(QSize)
    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('Activity')

        self.setWindowTitle(trans._('Activity'))

        activity_layout = QVBoxLayout()
        activity_layout.addStretch()

        self.setLayout(activity_layout)
