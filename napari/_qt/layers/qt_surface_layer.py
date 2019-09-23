from .qt_image_base_layer import QtBaseImageControls
from qtpy.QtWidgets import QLabel


class QtSurfaceControls(QtBaseImageControls):
    def __init__(self, layer):
        super().__init__(layer)

        self.grid_layout.addWidget(QLabel('opacity:'), 0, 0, 1, 3)
        self.grid_layout.addWidget(self.opacitySilder, 0, 3, 1, 4)
        self.grid_layout.addWidget(QLabel('contrast limits:'), 1, 0, 1, 3)
        self.grid_layout.addWidget(self.contrastLimitsSlider, 1, 3, 1, 4)
        self.grid_layout.addWidget(QLabel('colormap:'), 2, 0, 1, 3)
        self.grid_layout.addWidget(self.colormapComboBox, 2, 3, 1, 4)
        self.grid_layout.addWidget(QLabel('blending:'), 3, 0, 1, 3)
        self.grid_layout.addWidget(self.blendComboBox, 3, 3, 1, 4)
        self.grid_layout.setRowStretch(4, 1)
        self.grid_layout.setVerticalSpacing(4)
