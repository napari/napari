from __future__ import annotations

from typing import TYPE_CHECKING

from napari._qt.layer_controls.qt_image_controls_base import (
    QtBaseImageControls,
)
from napari._qt.layer_controls.widgets._surface import QtShadingComboBoxControl
from napari._qt.utils import set_widgets_enabled_with_opacity

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
    """

    layer: napari.layers.Surface
    PAN_ZOOM_ACTION_NAME = 'activate_surface_pan_zoom_mode'
    TRANSFORM_ACTION_NAME = 'activate_surface_transform_mode'

    def __init__(self, layer: napari.layers.Surface) -> None:
        super().__init__(layer)
        # Surface emits `data` when vertex_values or vertex_colors are reassigned.
        self.layer.events.data.connect(self._on_surface_coloring_change)

        # Setup widgets controls
        self._shading_combobox_control = QtShadingComboBoxControl(self, layer)
        self._add_widget_controls(self._shading_combobox_control)
        self._on_surface_coloring_change()

    def _on_surface_coloring_change(self) -> None:
        """Disable scalar-color controls when direct vertex colors are active."""
        enabled = self.layer.vertex_colors is None
        for control in (
            self._contrast_limits_control,
            self._gamma_slider_control,
            self._colormap_control,
        ):
            for label, widget in control.get_widget_controls():
                set_widgets_enabled_with_opacity(
                    self, (label, widget), enabled
                )
