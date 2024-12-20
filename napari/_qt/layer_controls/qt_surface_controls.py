from typing import TYPE_CHECKING

from napari._qt.layer_controls.qt_image_controls_base import (
    QtBaseImageControls,
)
from napari._qt.layer_controls.widgets._surface import QtShadingComboBoxControl

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
    MODE : Enum
        Available modes in the associated layer.
    PAN_ZOOM_ACTION_NAME : str
        String id for the pan-zoom action to bind to the pan_zoom button.
    TRANSFORM_ACTION_NAME : str
        String id for the transform action to bind to the transform button.
    autoScaleBar : qtpy.QtWidgets.QWidget
        Widget to wrap push buttons related with the layer auto-contrast funtionality.
    blendComboBox : qtpy.QtWidgets.QComboBox
        Dropdown widget to select blending mode of layer.
    blendLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the blending combobox widget.
    button_grid : qtpy.QtWidgets.QGridLayout
        GridLayout for the layer mode buttons
    button_group : qtpy.QtWidgets.QButtonGroup
        Button group for image based layer modes (PAN_ZOOM TRANSFORM).
    colormapWidget : qtpy.QtWidgets.QWidget
        Widget to wrap combobox and label widgets related with the layer colormap attribute.
    contrastLimitsSlider : superqt.QRangeSlider
        Contrast range slider widget.
    gammaSlider : qtpy.QtWidgets.QSlider
        Gamma adjustment slider widget.
    layer : napari.layers.Surface
        An instance of a napari Surface layer.
    opacityLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the opacity slider widget.
    opacitySlider : qtpy.QtWidgets.QSlider
        Slider controlling opacity of the layer.
    panzoom_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to pan/zoom shapes layer.
    shadingComboBox : qtpy.QtWidgets.QComboBox
        ComboBox controlling current shading value of the layer.
    shadingComboBoxLabel : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the shading value chooser widget.
    transform_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to transform shapes layer.
    """

    layer: 'napari.layers.Surface'
    PAN_ZOOM_ACTION_NAME = 'activate_surface_pan_zoom_mode'
    TRANSFORM_ACTION_NAME = 'activate_surface_transform_mode'

    def __init__(self, layer) -> None:
        super().__init__(layer)

        # Setup widgets controls
        self._add_widget_controls(QtShadingComboBoxControl(self, layer))
