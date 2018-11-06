from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QSlider


class QtControls(QHBoxLayout):
    def __init__(self):
        super().__init__()

        self.addWidget(QSlider(Qt.Vertical))
        self.addWidget(QSlider(Qt.Vertical))
