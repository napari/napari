from qtpy.QtWidgets import QComboBox, QHBoxLayout

from ...layers.surface._surface_constants import SHADING_TRANSLATION
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
        self._populate_grid(
            ('opacity:', self.opacitySlider),
            ('contrast limits:', self.contrastLimitsSlider),
            ('auto-contrast:', self.autoScaleBar),
            ('gamma:', self.gammaSlider),
            ('colormap:', colormap_layout),
            ('blending:', self.blendComboBox),
            ('shading:', self.shadingComboBox),
        )

    def changeShading(self, text):
        """Change shading value on the surface layer.
        Parameters
        ----------
        text : str
            Name of shading mode, eg: 'flat', 'smooth', 'none'.
        """
        self.layer.shading = self.shadingComboBox.currentData()
