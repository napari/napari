from math import sqrt
from PyQt5.QtWidgets import QHBoxLayout, QWidget

from ...util.range_slider import QVRangeSlider


class QtControls(QWidget):
    def __init__(self, clim_min, clim_max):
        super().__init__()

        self.hbox_layout = QHBoxLayout()

        # Gamma Slider
        self.gammaSlider = QVRangeSlider(slider_range=[-5.0, 5.0, 0.02], values=[-2.5, 2.5])
        self.gammaSlider.setEmitWhileMoving(True)
        self.hbox_layout.addWidget(self.gammaSlider)

        # Clim Slider
        self.climSlider = QVRangeSlider(slider_range=[clim_min, clim_max, sqrt(clim_max-clim_min)], values=[clim_min*1.1+1, clim_max*0.9-1])  # TODO: decide how to choose step for slider
        self.climSlider.setEmitWhileMoving(True)
        self.hbox_layout.addWidget(self.climSlider)

        self.setLayout(self.hbox_layout)
