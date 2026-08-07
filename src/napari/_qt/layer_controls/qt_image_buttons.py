from __future__ import annotations

from typing import TYPE_CHECKING

from napari._qt.layer_controls.qt_layer_buttons_base import QtLayerButtons

if TYPE_CHECKING:
    import napari.layers


class QtImageButtons(QtLayerButtons):
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

    layer: napari.layers.Image
    PAN_ZOOM_ACTION_NAME = 'activate_image_pan_zoom_mode'
    TRANSFORM_ACTION_NAME = 'activate_image_transform_mode'

    def __init__(self, layer) -> None:
        super().__init__(layer)
        # Setup widgets controls
        self._on_ndisplay_changed()

    def _on_ndisplay_changed(self):
        """Update widget visibility based on 2D and 3D visualization modes."""
        super()._on_ndisplay_changed()
