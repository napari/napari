"""Provides an Activity Dock that allows the user to view running progress bars
"""
from qtpy.QtWidgets import QVBoxLayout, QWidget


class QtActivityDock(QWidget):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)
