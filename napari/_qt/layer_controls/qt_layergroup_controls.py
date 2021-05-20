from qtpy.QtWidgets import QLabel

from ...utils.translations import trans
from .qt_layer_controls_base import QtLayerControls


class QtLayergroupControls(QtLayerControls):
    def __init__(self, layer):
        super().__init__(layer)

        self.grid_layout.addWidget(QLabel(trans._('opacity:')), 0, 0)
        self.grid_layout.addWidget(self.opacitySlider, 0, 1)
        self.grid_layout.addWidget(QLabel(trans._('blending:')), 1, 0)
        self.grid_layout.addWidget(self.blendComboBox, 1, 1)
        self.grid_layout.setRowStretch(9, 1)
        self.grid_layout.setColumnStretch(1, 1)
        self.grid_layout.setSpacing(4)
