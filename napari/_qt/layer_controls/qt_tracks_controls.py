from typing import TYPE_CHECKING

from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari._qt.layer_controls.widgets import (
    QtColorPropertiesComboBoxControl,
    QtGraphCheckBoxControl,
    QtHeadLengthSliderControl,
    QtIdCheckBoxControl,
    QtSimpleColormapComboBoxControl,
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
    layer : layers.Tracks
        An instance of a Tracks layer.
    button_group : qtpy.QtWidgets.QButtonGroup
        Button group of points layer modes (ADD, PAN_ZOOM, SELECT).
    panzoom_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button for pan/zoom mode.
    transform_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select transform mode.

    """

    layer: 'napari.layers.Tracks'
    MODE = Mode
    PAN_ZOOM_ACTION_NAME = 'activate_tracks_pan_zoom_mode'
    TRANSFORM_ACTION_NAME = 'activate_tracks_transform_mode'

    def __init__(self, layer) -> None:
        super().__init__(layer)
        # Setup widgets controls
        self._add_widget_controls(
            QtColorPropertiesComboBoxControl(self, layer)
        )
        self._add_widget_controls(QtSimpleColormapComboBoxControl(self, layer))
        self._add_widget_controls(QtTailWidthSliderControl(self, layer))
        self._add_widget_controls(QtTailLengthSliderControl(self, layer))
        self._add_widget_controls(QtHeadLengthSliderControl(self, layer))
        self._add_widget_controls(QtTailDisplayCheckBoxControl(self, layer))
        self._add_widget_controls(QtIdCheckBoxControl(self, layer))
        self._add_widget_controls(QtGraphCheckBoxControl(self, layer))
