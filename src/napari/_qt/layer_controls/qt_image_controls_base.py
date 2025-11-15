from __future__ import annotations

from typing import TYPE_CHECKING

from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari._qt.layer_controls.widgets import (
    QtColormapControl,
    QtContrastLimitsControl,
    QtGammaSliderControl,
)
from napari._qt.layer_controls.widgets.qt_histogram_control import (
    QtHistogramControl,
)

if TYPE_CHECKING:
    from napari.layers import Image


class QtBaseImageControls(QtLayerControls):
    """Superclass for classes requiring colormaps, contrast & gamma sliders.

    This class is never directly instantiated anywhere.
    It is subclassed by QtImageControls and QtSurfaceControls.

    Parameters
    ----------
    layer : napari.layers.Image | napari.layers.Surface
        An instance of a napari Image or Surface layer.

    Attributes
    ----------
    _colormap_control : napari._qt.layer_controls.widgets.QtColormapControl
        Widget that wraps combobox and label widgets related with the layer colormap attribute.
    _contrast_limits_control : napari._qt.layer_controls.widgets.QtContrastLimitsControl
        Widget that wraps layer contrast range slider and autocontrast widgets.
    _gamma_slider_control : napari._qt.layer_controls.widgets.QtGammaSliderControl
        Widget that wraps layer gamma adjustment slider widget.
    _histogram_control : napari._qt.layer_controls.widgets._image.QtHistogramControl
        Collapsible widget that shows layer histogram with visualization and settings.
    """

    def __init__(self, layer: Image) -> None:
        super().__init__(layer)
        # Setup widgets controls
        self._contrast_limits_control = QtContrastLimitsControl(self, layer)
        self._add_widget_controls(self._contrast_limits_control)
        self._gamma_slider_control = QtGammaSliderControl(self, layer)
        self._add_widget_controls(self._gamma_slider_control)
        self._histogram_control = QtHistogramControl(self, layer)
        self._add_widget_controls(self._histogram_control)
        self._colormap_control = QtColormapControl(self, layer)
        self._add_widget_controls(self._colormap_control)
