"""Provides a ProgressBar dock that allows the user to view running progress bars
"""
import typing

from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

from ...utils.translations import trans


class QtProgressBarDock(QWidget):
    def __init__(self, parent: typing.Optional['QWidget']) -> None:
        super().__init__(parent=parent)
        title = QLabel(trans._('Progress Bar Dock'))
        title.setObjectName("h3")

        layout = QVBoxLayout(self)
        layout.addWidget(title)
