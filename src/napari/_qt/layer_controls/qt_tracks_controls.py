from typing import TYPE_CHECKING

from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari._qt.layer_controls.widgets._tracks import (
    QtColormapComboBoxControl,
    QtColorPropertiesComboBoxControl,
    QtGraphCheckBoxControl,
    QtHeadLengthSliderControl,
    QtHideCompletedTracksCheckBoxControl,
    QtIdCheckBoxControl,
    QtTailDisplayCheckBoxControl,
    QtTailLengthSliderControl,
    QtTailWidthSliderControl,
)
from napari.layers.base._base_constants import Mode

if TYPE_CHECKING:
    import napari.layers


class QtTracksControls(QtLayerControls):
    """Qt view and controls for the Tracks layer.

    Parameters
    ----------
    layer : napari.layers.Tracks
        An instance of a Tracks layer.

    Attributes
    ----------
    _color_properties_combobox_control : napari._qt.layer_controls.widgets._tracks.QtColorPropertiesComboBoxControl
        Widget that wraps a comboBox controlling the layer color properties.
    _colormap_combobox_control : napari._qt.layer_controls.widgets._tracks.QtColormapComboBoxControl
        Widget that wraps a comboBox controlling current colormap of the layer.
    _graph_checkbox_control : napari._qt.layer_controls.widgets._tracks.QtGraphCheckBoxControl
        Checkbox controlling if graph of the layer should be shown.
    _head_length_slider_control : napari._qt.layer_controls.widgets._tracks.QtHeadLengthSliderControl
        Widget that wraps a slider controlling head length of the layer.
    _id_checkbox_control : napari._qt.layer_controls.widgets._tracks.QtIdCheckBoxControl
        Widget that wraps a checkbox controlling if id of the layer should be shown.
    _tail_display_checkbox_control : napari._qt.layer_controls.widgets._tracks.QtTailDisplayCheckBoxControl
        Widget that wraps a checkbox controlling if tails of the layer should be shown.
    _tail_length_slider_control : napari._qt.layer_controls.widgets._tracks.QtTailLengthSliderControl
        Widget that wraps a slider controlling tail length of the layer.
    _tail_width_slider_control : napari._qt.layer_controls.widgets._tracks.QtTailWidthSliderControl
        Widget that wraps a slider controlling tail width of the layer.
    _hide_completed_tracks_checkbox_control : napari._qt.layer_controls.widgets._tracks.QtHideCompletedTracksCheckBoxControl
        Widget that wraps a checkbox controlling if completed tracks of the layer should be hidden.
    """

    layer: 'napari.layers.Tracks'
    MODE = Mode
    PAN_ZOOM_ACTION_NAME = 'activate_tracks_pan_zoom_mode'
    TRANSFORM_ACTION_NAME = 'activate_tracks_transform_mode'

    def __init__(self, layer) -> None:
        super().__init__(layer)
        # Setup widgets controls
        self._color_properties_combobox_control = (
            QtColorPropertiesComboBoxControl(self, layer)
        )
        self._add_widget_controls(self._color_properties_combobox_control)
        self._colormap_combobox_control = QtColormapComboBoxControl(
            self, layer
        )
        self._add_widget_controls(self._colormap_combobox_control)
        self._tail_width_slider_control = QtTailWidthSliderControl(self, layer)
        self._add_widget_controls(self._tail_width_slider_control)
        self._tail_length_slider_control = QtTailLengthSliderControl(
            self, layer
        )
        self._add_widget_controls(self._tail_length_slider_control)
        self._head_length_slider_control = QtHeadLengthSliderControl(
            self, layer
        )
        self._add_widget_controls(self._head_length_slider_control)
        self._tail_display_checkbox_control = QtTailDisplayCheckBoxControl(
            self, layer
        )
        self._add_widget_controls(self._tail_display_checkbox_control)
        self._id_checkbox_control = QtIdCheckBoxControl(self, layer)
        self._add_widget_controls(self._id_checkbox_control)
        self._graph_checkbox_control = QtGraphCheckBoxControl(self, layer)
        self._add_widget_controls(self._graph_checkbox_control)
        self._hide_completed_tracks_checkbox_control = (
            QtHideCompletedTracksCheckBoxControl(self, layer)
        )
        self._add_widget_controls(self._hide_completed_tracks_checkbox_control)
