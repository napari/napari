from PyQt5.QtWidgets import QHBoxLayout, QWidget
from PyQt5.QtCore import QSize

from ...util.range_slider import QVRangeSlider


class QtControlBars(QWidget):
    def __init__(self):
        super().__init__()

        self.hbox_layout = QHBoxLayout()

        # Gamma Slider
        # self.gammaSlider = QVRangeSlider(slider_range=[-5.0, 5.0, 0.02], values=[-2.5, 2.5])
        # self.gammaSlider.setEmitWhileMoving(True)
        # self.hbox_layout.addWidget(self.gammaSlider)

        # Clim Slider
        self.climSlider = QVRangeSlider(slider_range=[0, 1, 0.0001], values=[0, 1], parent=self)
        self.climSlider.setEmitWhileMoving(True)
        self.climSlider.collapsable = False
        self.hbox_layout.addWidget(self.climSlider)

        self.setLayout(self.hbox_layout)
        self.setMouseTracking(True)
        self.setMinimumSize(QSize(40, 40))
