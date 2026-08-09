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
