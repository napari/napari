from typing import TYPE_CHECKING

from qtpy.QtWidgets import QComboBox, QHBoxLayout

from napari._qt.layer_controls.qt_image_controls_base import (
    QtBaseImageControls,
)
from napari.layers.surface._surface_constants import SHADING_TRANSLATION
from napari.utils.translations import trans

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
    MODE : Enum
        Available modes in the associated layer.
    PAN_ZOOM_ACTION_NAME : str
        String id for the pan-zoom action to bind to the pan_zoom button.
    TRANSFORM_ACTION_NAME : str
        String id for the transform action to bind to the transform button.
    button_group : qtpy.QtWidgets.QButtonGroup
        Button group for image based layer modes (PAN_ZOOM TRANSFORM).
    button_grid : qtpy.QtWidgets.QGridLayout
        GridLayout for the layer mode buttons
    panzoom_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to pan/zoom shapes layer.
    transform_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to transform shapes layer.

    """

    layer: 'napari.layers.Surface'
    PAN_ZOOM_ACTION_NAME = 'activate_surface_pan_zoom_mode'
    TRANSFORM_ACTION_NAME = 'activate_surface_transform_mode'

    def __init__(self, layer) -> None:
        super().__init__(layer)

        self.layer.events.shading.connect(self._on_shading_change)

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

        self.layout().addRow(self.button_grid)
        self.layout().addRow(self.opacityLabel, self.opacitySlider)
        self.layout().addRow(trans._('blending:'), self.blendComboBox)
        self.layout().addRow(
            trans._('contrast limits:'), self.contrastLimitsSlider
        )
        self.layout().addRow(trans._('auto-contrast:'), self.autoScaleBar)
        self.layout().addRow(trans._('gamma:'), self.gammaSlider)
        self.layout().addRow(trans._('colormap:'), colormap_layout)
        self.layout().addRow(trans._('shading:'), self.shadingComboBox)

    def changeShading(self, text):
        """Change shading value on the surface layer.
        Parameters
        ----------
        text : str
            Name of shading mode, eg: 'flat', 'smooth', 'none'.
        """
        self.layer.shading = self.shadingComboBox.currentData()

    def _on_shading_change(self):
        """Receive layer model shading change event and update combobox."""
        with self.layer.events.shading.blocker():
            self.shadingComboBox.setCurrentIndex(
                self.shadingComboBox.findData(
                    SHADING_TRANSLATION[self.layer.shading]
                )
            )
