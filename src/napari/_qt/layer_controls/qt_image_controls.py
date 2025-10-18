from typing import TYPE_CHECKING

from napari._qt.layer_controls.qt_image_controls_base import (
    QtBaseImageControls,
)
from napari._qt.layer_controls.widgets import QtProjectionModeControl
from napari._qt.layer_controls.widgets._image import (
    QtDepictionControl,
    QtImageRenderControl,
    QtInterpolationComboBoxControl,
)

if TYPE_CHECKING:
    import napari.layers


class QtImageControls(QtBaseImageControls):
    """Qt view and controls for the napari Image layer.

    Parameters
    ----------
    layer : napari.layers.Image
        An instance of a napari Image layer.

    Attributes
    ----------
    _depiction_control : napari._qt.layer_controls.widgets._image.QtDepictionControl
        Widget that wraps widgets related with the layer depiction and plane attributes.
    _interpolation_control : napari._qt.layer_controls.widgets._image.QtInterpolationComboBoxControl
        Widget that wraps dropdown menu to select the interpolation mode for image display.
    _projection_mode_control : napari._qt.layer_controls.widgets.QtProjectionModeControl
        Widget that wraps dropdown menu to select the projection mode for the layer.
    _render_control : napari._qt.layer_controls.widgets._image.QtImageRenderControl
        Widget that wraps widgets related with the method used to render the layer.
    """

    layer: 'napari.layers.Image'
    PAN_ZOOM_ACTION_NAME = 'activate_image_pan_zoom_mode'
    TRANSFORM_ACTION_NAME = 'activate_image_transform_mode'

    def __init__(self, layer) -> None:
        super().__init__(layer)
        # Setup widgets controls
        self._projection_mode_control = QtProjectionModeControl(self, layer)
        self._add_widget_controls(self._projection_mode_control)
        self._interpolation_control = QtInterpolationComboBoxControl(
            self, layer
        )
        self._add_widget_controls(self._interpolation_control)
        self._depiction_control = QtDepictionControl(self, layer)
        self._add_widget_controls(self._depiction_control)
        self._render_control = QtImageRenderControl(self, layer)
        self._add_widget_controls(self._render_control)

        self._on_ndisplay_changed()

    def _on_ndisplay_changed(self):
        """Update widget visibility based on 2D and 3D visualization modes."""
        self._interpolation_control._update_interpolation_combo(self.ndisplay)
        self._depiction_control._update_plane_parameter_visibility()
        if self.ndisplay == 2:
            self._render_control._on_display_change_hide()
            self._depiction_control._on_display_change_hide()
        else:
            self._render_control._on_display_change_show()
            self._depiction_control._on_display_change_show()
        super()._on_ndisplay_changed()
