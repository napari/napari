"""Provides a ProgressBar dock that allows the user to view running progress bars
"""
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

from ...utils.translations import trans


class QtActivityDock(QWidget):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        title = QLabel(trans._('Progress Bar Dock'))
        title.setObjectName("h3")

        layout = QVBoxLayout(self)
        layout.addWidget(title)

        self.layout = layout
