from qtpy.QtWidgets import QHBoxLayout, QLabel

from ...utils.translations import trans
from .qt_image_controls_base import QtBaseImageControls


class QtSurfaceControls(QtBaseImageControls):
    """Qt view and controls for the napari Surface layer.

    Parameters
    ----------
    layer : napari.layers.Surface
        An instance of a napari Surface layer.

    Attributes
    ----------
    grid_layout : qtpy.QtWidgets.QGridLayout
        Layout of Qt widget controls for the layer.
    layer : napari.layers.Surface
        An instance of a napari Surface layer.

    """

    def __init__(self, layer):
        super().__init__(layer)

        colormap_layout = QHBoxLayout()
        colormap_layout.addWidget(self.colorbarLabel)
        colormap_layout.addWidget(self.colormapComboBox)
        colormap_layout.addStretch(1)

        # grid_layout created in QtLayerControls
        # addWidget(widget, row, column, [row_span, column_span])
        self.grid_layout.addWidget(QLabel(trans._('opacity:')), 0, 0)
        self.grid_layout.addWidget(self.opacitySlider, 0, 1)
        self.grid_layout.addWidget(QLabel(trans._('contrast limits:')), 1, 0)
        self.grid_layout.addWidget(self.contrastLimitsSlider, 1, 1)
        self.grid_layout.addWidget(QLabel(trans._('gamma:')), 2, 0)
        self.grid_layout.addWidget(self.gammaSlider, 2, 1)
        self.grid_layout.addWidget(QLabel(trans._('colormap:')), 3, 0)
        self.grid_layout.addLayout(colormap_layout, 3, 1)
        self.grid_layout.addWidget(QLabel(trans._('blending:')), 4, 0)
        self.grid_layout.addWidget(self.blendComboBox, 4, 1)
        self.grid_layout.setRowStretch(5, 1)
        self.grid_layout.setColumnStretch(1, 1)
        self.grid_layout.setSpacing(4)
