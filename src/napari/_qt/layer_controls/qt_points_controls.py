from typing import TYPE_CHECKING

from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari._qt.layer_controls.widgets import (
    QtFaceColorControl,
    QtOutSliceCheckBoxControl,
    QtProjectionModeControl,
    QtTextVisibilityControl,
)
from napari._qt.layer_controls.widgets._points import (
    QtBorderColorControl,
    QtCurrentSizeSliderControl,
    QtSymbolComboBoxControl,
)
from napari._qt.widgets.qt_mode_buttons import QtModePushButton
from napari.layers.points._points_constants import Mode
from napari.utils.action_manager import action_manager
from napari.utils.events import disconnect_events
from napari.utils.translations import trans

if TYPE_CHECKING:
    import napari.layers


class QtPointsControls(QtLayerControls):
    """Qt view and controls for the napari Points layer.

    Parameters
    ----------
    layer : napari.layers.Points
        An instance of a napari Points layer.

    Attributes
    ----------
    _border_color_control : napari._qt.layer_controls.widgets._points.QtBorderColorControl
        Widget to handle point's border color.
    _current_size_slider_control : napari._qt.layer_controls.widgets._points.QtCurrentSizeSliderControl
        Widget that wraps slider controlling size of points.
    _face_color_control : napari._qt.layer_controls.widgets.QtFaceColorControl
        Widget to select display color for points faces.
    _out_slice_checkbox_control : napari._qt.layer_controls.widgets.QtOutSliceCheckBoxControl
        Widget that wraps a checkbox to indicate whether to render out of slice.
    _projection_mode_control : napari._qt.layer_controls.widgets.QtProjectionModeControl
        Widget that wraps dropdown menu to select the projection mode for the layer.
    _symbol_combobox_control : napari._qt.layer_controls.widgets._points.QtSymbolComboBoxControl
        Widget that wraps a dropdown list of symbol options for points markers.
    _text_visibility_control : napari._qt.layer_controls.widgets.QtTextVisibilityControl
        Widget that wraps a checkbox controlling if text on the layer is visible or not.
    addition_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to add points to layer.
    delete_button : qtpy.QtWidgets.QtModePushButton
        Button to delete points from layer.
    select_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select points from layer.

    Raises
    ------
    ValueError
        Raise error if points mode is not recognized.
        Points mode must be one of: ADD, PAN_ZOOM, or SELECT.
    """

    layer: 'napari.layers.Points'
    MODE = Mode
    PAN_ZOOM_ACTION_NAME = 'activate_points_pan_zoom_mode'
    TRANSFORM_ACTION_NAME = 'activate_points_transform_mode'

    def __init__(self, layer) -> None:
        super().__init__(layer)

        # Setup buttons
        self.select_button = self._radio_button(
            layer,
            'select_points',
            Mode.SELECT,
            True,
            'activate_points_select_mode',
        )
        self.addition_button = self._radio_button(
            layer,
            'add_points',
            Mode.ADD,
            True,
            'activate_points_add_mode',
        )
        self.delete_button = QtModePushButton(
            layer,
            'delete_shape',
        )
        action_manager.bind_button(
            'napari:delete_selected_points', self.delete_button
        )
        self._EDIT_BUTTONS += (self.delete_button,)
        self._on_editable_or_visible_change()

        self.button_grid.addWidget(self.delete_button, 0, 3)
        self.button_grid.addWidget(self.addition_button, 0, 4)
        self.button_grid.addWidget(self.select_button, 0, 5)

        # Setup widgets controls
        self._projection_mode_control = QtProjectionModeControl(self, layer)
        self._add_widget_controls(self._projection_mode_control)
        self._current_size_slider_control = QtCurrentSizeSliderControl(
            self, layer
        )
        self._add_widget_controls(self._current_size_slider_control)
        self._symbol_combobox_control = QtSymbolComboBoxControl(self, layer)
        self._add_widget_controls(self._symbol_combobox_control)
        self._face_color_control = QtFaceColorControl(
            self,
            layer,
            tooltip=trans._(
                'Click to set the face color of currently selected points and any added afterwards.'
            ),
        )
        self._add_widget_controls(self._face_color_control)
        self._border_color_control = QtBorderColorControl(self, layer)
        self._add_widget_controls(self._border_color_control)
        self._text_visibility_control = QtTextVisibilityControl(self, layer)
        self._add_widget_controls(self._text_visibility_control)
        self._out_slice_checkbox_control = QtOutSliceCheckBoxControl(
            self, layer
        )
        self._add_widget_controls(self._out_slice_checkbox_control)

    def _on_mode_change(self, event):
        """Update ticks in checkbox widgets when points layer mode is changed.

        Available modes for points layer are:
        * ADD
        * SELECT
        * PAN_ZOOM
        * TRANSFORM

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.

        Raises
        ------
        ValueError
            Raise error if event.mode is not ADD, PAN_ZOOM, TRANSFORM or SELECT.
        """
        super()._on_mode_change(event)

    def _on_ndisplay_changed(self):
        self.layer.editable = not (self.layer.ndim == 2 and self.ndisplay == 3)
        super()._on_ndisplay_changed()

    def close(self):
        """Disconnect events when widget is closing."""
        disconnect_events(self.layer.text.events, self)
        super().close()
