from .qt_image_base_layer import QtBaseImageControls
from qtpy.QtWidgets import QLabel, QHBoxLayout


class QtSurfaceControls(QtBaseImageControls):
    def __init__(self, layer):
        super().__init__(layer)

        colormap_layout = QHBoxLayout()
        colormap_layout.addWidget(self.colorbarLabel)
        colormap_layout.addWidget(self.colormapComboBox)
        colormap_layout.addStretch(1)

        # grid_layout created in QtLayerControls
        # addWidget(widget, row, column, [row_span, column_span])
        self.grid_layout.addWidget(QLabel('opacity:'), 0, 0)
        self.grid_layout.addWidget(self.opacitySlider, 0, 1)
        self.grid_layout.addWidget(QLabel('contrast limits:'), 1, 0)
        self.grid_layout.addWidget(self.contrastLimitsSlider, 1, 1)
        self.grid_layout.addWidget(QLabel('gamma:'), 2, 0)
        self.grid_layout.addWidget(self.gammaSlider, 2, 1)
        self.grid_layout.addWidget(QLabel('colormap:'), 3, 0)
        self.grid_layout.addLayout(colormap_layout, 3, 1)
        self.grid_layout.addWidget(QLabel('blending:'), 4, 0)
        self.grid_layout.addWidget(self.blendComboBox, 4, 1)
        self.grid_layout.setRowStretch(5, 1)
        self.grid_layout.setColumnStretch(1, 1)
        self.grid_layout.setSpacing(4)
