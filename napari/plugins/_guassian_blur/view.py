from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QComboBox, QSlider, QCheckBox
import numpy as np

from .._base_plugin import QtPlugin


class QtGaussianBlur(QtPlugin):
    def __init__(self, plugin):
        super().__init__(plugin)

        sld = QSlider(Qt.Horizontal, self)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setFixedWidth(75)
        sld.setMinimum(0)
        sld.setMaximum(100)
        sld.setSingleStep(1)
        sld.setValue(int(self.plugin.blur*10))
        sld.valueChanged[int].connect(lambda value=sld: self.blur(value))
        self.grid_layout.addWidget(QLabel('blur:'), 1, 0)
        self.grid_layout.addWidget(sld, 1, 1)

    def blur(self, value):
        self.plugin.blur = value/10
