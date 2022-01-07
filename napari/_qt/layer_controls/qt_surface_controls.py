from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel

from ...layers.surface._surface_constants import SHADING_TRANSLATION
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

        shading_comboBox = QComboBox(self)
        for display_name, shading in SHADING_TRANSLATION.items():
            shading_comboBox.addItem(display_name, shading)
        index = shading_comboBox.findData(
            SHADING_TRANSLATION[self.layer.shading]
        )
        shading_comboBox.setCurrentIndex(index)
        shading_comboBox.activated[str].connect(self.changeShading)
        self.shadingComboBox = shading_comboBox

        # grid_layout created in QtLayerControls
        # addWidget(widget, row, column, [row_span, column_span])
        self.grid_layout.addWidget(QLabel(trans._('opacity:')), 0, 0)
        self.grid_layout.addWidget(self.opacitySlider, 0, 1)
        self.grid_layout.addWidget(QLabel(trans._('contrast limits:')), 1, 0)
        self.grid_layout.addWidget(self.contrastLimitsSlider, 1, 1)
        self.grid_layout.addWidget(QLabel(trans._('auto-contrast:')), 2, 0)
        self.grid_layout.addWidget(self.autoScaleBar, 2, 1)
        self.grid_layout.addWidget(QLabel(trans._('gamma:')), 3, 0)
        self.grid_layout.addWidget(self.gammaSlider, 3, 1)
        self.grid_layout.addWidget(QLabel(trans._('colormap:')), 4, 0)
        self.grid_layout.addLayout(colormap_layout, 4, 1)
        self.grid_layout.addWidget(QLabel(trans._('blending:')), 5, 0)
        self.grid_layout.addWidget(self.blendComboBox, 5, 1)
        self.grid_layout.addWidget(QLabel(trans._('shading:')), 6, 0)
        self.grid_layout.addWidget(self.shadingComboBox, 6, 1)

        self.grid_layout.setRowStretch(7, 1)
        self.grid_layout.setColumnStretch(1, 1)
        self.grid_layout.setSpacing(4)

    def changeShading(self, text):
        """Change shading value on the surface layer.
        Parameters
        ----------
        text : str
            Name of shading mode, eg: 'flat', 'smooth', 'none'.
        """
        self.layer.shading = self.shadingComboBox.currentData()
