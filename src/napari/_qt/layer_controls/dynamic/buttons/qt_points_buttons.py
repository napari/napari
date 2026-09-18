from __future__ import annotations

from typing import TYPE_CHECKING

from napari._qt.layer_controls.dynamic.buttons.qt_layer_buttons_base import (
    QtLayerButtons,
)
from napari._qt.widgets.qt_mode_buttons import QtModePushButton
from napari.layers.points._points_constants import Mode
from napari.utils.action_manager import action_manager

if TYPE_CHECKING:
    import napari.layers


class QtPointsButtons(QtLayerButtons):
    """Qt view and controls for the napari Points layer.

    Parameters
    ----------
    layer : napari.layers.Points
        An instance of a napari Points layer.

    Attributes
    ----------
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

    layer: napari.layers.Points
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

        self.addWidget(self.delete_button, 0, 3)
        self.addWidget(self.addition_button, 0, 4)
        self.addWidget(self.select_button, 0, 5)

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
