from PyQt5.QtWidgets import QHBoxLayout, QWidget
from PyQt5.QtCore import QSize

from ....util.range_slider import QVRangeSlider


class QtImageControls(QWidget):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer

        self.setMouseTracking(True)

    def mouseMoveEvent(self, event):
        self.layer.status = self.layer._clim_msg
