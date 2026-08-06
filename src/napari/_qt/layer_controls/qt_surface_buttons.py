from __future__ import annotations

from typing import TYPE_CHECKING

from napari._qt.layer_controls.qt_layer_buttons_base import QtLayerButtons

if TYPE_CHECKING:
    import napari.layers


class QtSurfaceButtons(QtLayerButtons):
    """Qt view and controls for the napari Surface layer.

    Parameters
    ----------
    layer : napari.layers.Surface
        An instance of a napari Surface layer.

    Attributes
    ----------
    _shading_combobox_control : napari._qt.layer_controls.widgets._surface.QtShadingComboBoxControl
        Widget that wraps comboBox controlling current shading value of the layer.
    """

    PAN_ZOOM_ACTION_NAME = 'activate_surface_pan_zoom_mode'
    TRANSFORM_ACTION_NAME = 'activate_surface_transform_mode'

    def __init__(self, layer: napari.layers.Surface) -> None:
        super().__init__(layer)
