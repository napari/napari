from typing import TYPE_CHECKING

from napari._qt.layer_controls.qt_image_controls_base import (
    QtBaseImageControls,
)
from napari._qt.layer_controls.widgets import (
    QtProjectionModeControl,
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
    _shading_combobox_control : napari._qt.layer_controls.widgets._surface.QtShadingComboBoxControl
        Widget that wraps comboBox controlling current shading value of the layer.
    _projection_mode_control : napari._qt.layer_controls.widgets.QtProjectionModeControl
        Widget that wraps dropdown menu to select the projection mode for the layer.
    """

    layer: 'napari.layers.Surface'
    PAN_ZOOM_ACTION_NAME = 'activate_surface_pan_zoom_mode'
    TRANSFORM_ACTION_NAME = 'activate_surface_transform_mode'

    def __init__(self, layer) -> None:
        super().__init__(layer)

        # Setup widgets controls
        self._shading_combobox_control = QtShadingComboBoxControl(self, layer)
        self._add_widget_controls(self._shading_combobox_control)
        self._projection_mode_control = QtProjectionModeControl(self, layer)
        self._add_widget_controls(self._projection_mode_control)
