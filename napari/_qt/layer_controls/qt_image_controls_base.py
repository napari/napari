from __future__ import annotations

from typing import TYPE_CHECKING

from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari._qt.layer_controls.widgets import (
    QtAutoScaleControl,
    QtColormapControl,
    QtContrastLimitsSliderControl,
    QtGammaSliderControl,
)

if TYPE_CHECKING:
    from napari.layers import Image


class QtBaseImageControls(QtLayerControls):
    """Superclass for classes requiring colormaps, contrast & gamma sliders.

    This class is never directly instantiated anywhere.
    It is subclassed by QtImageControls and QtSurfaceControls.

    Parameters
    ----------
    layer : napari.layers.Layer
        An instance of a napari layer.

    Attributes
    ----------
    MODE : Enum
        Available modes in the associated layer.
    PAN_ZOOM_ACTION_NAME : str
        String id for the pan-zoom action to bind to the pan_zoom button.
    TRANSFORM_ACTION_NAME : str
        String id for the transform action to bind to the transform button.
    _auto_scale_control : napari._qt.layer_controls.widgets.QtAutoScaleControl
        Widget to wrap widgets related with the layer auto-contrast functionality.
    _colormap_control : napari._qt.layer_controls.widgets.QtColormapControl
        Widget to wrap combobox and label widgets related with the layer colormap attribute.
    _contrast_limits_slider_control : napari._qt.layer_controls.widgets.QtContrastLimitsSliderControl
        Widget to wrap layer contrast range slider widget.
    _gamma_slider_control : napari._qt.layer_controls.widgets.QtGammaSliderControl
        Widget to wrap layer gamma adjustment slider widget.
    button_grid : qtpy.QtWidgets.QGridLayout
        GridLayout for the layer mode buttons
    button_group : qtpy.QtWidgets.QButtonGroup
        Button group for image based layer modes (PAN_ZOOM TRANSFORM).
    layer : napari.layers.Layer
        An instance of a napari layer.
    panzoom_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to activate move camera mode for layer.
    transform_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to transform image layer.
    """

    def __init__(self, layer: Image) -> None:
        super().__init__(layer)
        # Setup widgets controls
        self._contrast_limits_slider_control = QtContrastLimitsSliderControl(
            self, layer
        )
        self._add_widget_controls(self._contrast_limits_slider_control)
        self._auto_scale_control = QtAutoScaleControl(self, layer)
        self._add_widget_controls(self._auto_scale_control)
        self._gamma_slider_control = QtGammaSliderControl(self, layer)
        self._add_widget_controls(self._gamma_slider_control)
        self._colormap_control = QtColormapControl(self, layer)
        self._add_widget_controls(self._colormap_control)
