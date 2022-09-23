from typing import TYPE_CHECKING

from qtpy.QtWidgets import QComboBox, QHBoxLayout

from ...layers.surface._surface_constants import SHADING_TRANSLATION
from ...utils.translations import trans
from .qt_image_controls_base import QtBaseImageControls

if TYPE_CHECKING:
    import napari.layers


class QtSurfaceControls(QtBaseImageControls):
    """Qt view and controls for the napari Surface layer.

    Parameters
    ----------
    layer : napari.layers.Surface
        An instance of a napari Surface layer.

    Attributes
    ----------
    layer : napari.layers.Surface
        An instance of a napari Surface layer.

    """

    layer: 'napari.layers.Surface'

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
        shading_comboBox.currentTextChanged.connect(self.changeShading)
        self.shadingComboBox = shading_comboBox

        self.layout().addRow(self.opacityLabel, self.opacitySlider)
        self.layout().addRow(
            trans._('contrast limits:'), self.contrastLimitsSlider
        )
        self.layout().addRow(trans._('auto-contrast:'), self.autoScaleBar)
        self.layout().addRow(trans._('gamma:'), self.gammaSlider)
        self.layout().addRow(trans._('colormap:'), colormap_layout)
        self.layout().addRow(trans._('blending:'), self.blendComboBox)
        self.layout().addRow(trans._('shading:'), self.shadingComboBox)

    def changeShading(self, text):
        """Change shading value on the surface layer.
        Parameters
        ----------
        text : str
            Name of shading mode, eg: 'flat', 'smooth', 'none'.
        """
        self.layer.shading = self.shadingComboBox.currentData()
