from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QWidget

from ...util.range_slider import QVRangeSlider


class QtControls(QWidget):
    def __init__(self):
        super().__init__()

        self.hbox_layout = QHBoxLayout()
        self.hbox_layout.addWidget(QVRangeSlider())
        self.hbox_layout.addWidget(QVRangeSlider())
        self.setLayout(self.hbox_layout)
