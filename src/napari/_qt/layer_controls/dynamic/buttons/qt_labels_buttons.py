from __future__ import annotations

from typing import TYPE_CHECKING

from napari._qt.layer_controls.dynamic.buttons.qt_layer_buttons_base import (
    QtLayerButtons,
)
from napari._qt.utils import set_widgets_enabled_with_opacity
from napari._qt.widgets.qt_mode_buttons import QtModePushButton
from napari.layers.labels._labels_constants import Mode

if TYPE_CHECKING:
    import napari.layers


INT32_MAX = 2**31 - 1


class QtLabelsButtons(QtLayerButtons):
    """Qt view and controls for the napari Labels layer.

    Parameters
    ----------
    layer : napari.layers.Labels
        An instance of a napari Labels layer.

    Attributes
    ----------
    colormap_update : napari._qt.widgets.qt_mode_buttons.QtModePushButton
        Button to update colormap of label layer.
    erase_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select ERASE mode on Labels layer.
    fill_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select FILL mode on Labels layer.
    paint_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select PAINT mode on Labels layer.
    pick_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select PICK mode on Labels layer.
    polygon_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select POLYGON mode on Labels layer.

    Raises
    ------
    ValueError
        Raise error if label mode is not PAN_ZOOM, PICKER, PAINT, ERASE, or
        FILL.
    """

    layer: napari.layers.Labels
    MODE = Mode
    PAN_ZOOM_ACTION_NAME = 'activate_labels_pan_zoom_mode'
    TRANSFORM_ACTION_NAME = 'activate_labels_transform_mode'

    def __init__(self, layer) -> None:
        super().__init__(layer)
        layer.events.n_edit_dimensions.connect(self._set_polygon_tool_state)

        # Setup buttons
        # shuffle colormap button
        self.colormap_update = QtModePushButton(
            layer,
            'shuffle',
            slot=self.change_color,
            tooltip='Shuffle colors',
        )

        self.pick_button = self._radio_button(
            layer,
            'picker',
            Mode.PICK,
            True,
            'activate_labels_picker_mode',
        )
        self.paint_button = self._radio_button(
            layer,
            'paint',
            Mode.PAINT,
            True,
            'activate_labels_paint_mode',
        )
        self.polygon_button = self._radio_button(
            layer,
            'labels_polygon',
            Mode.POLYGON,
            True,
            'activate_labels_polygon_mode',
        )
        self.fill_button = self._radio_button(
            layer,
            'fill',
            Mode.FILL,
            True,
            'activate_labels_fill_mode',
        )
        self.erase_button = self._radio_button(
            layer,
            'erase',
            Mode.ERASE,
            True,
            'activate_labels_erase_mode',
        )
        # don't bind with action manager as this would remove "Toggle with {shortcut}"
        self._on_editable_or_visible_change()

        self.addWidget(self.colormap_update, 0, 0)
        self.addWidget(self.erase_button, 0, 1)
        self.addWidget(self.paint_button, 0, 2)
        self.addWidget(self.polygon_button, 0, 3)
        self.addWidget(self.fill_button, 0, 4)
        self.addWidget(self.pick_button, 0, 5)

        self._on_ndisplay_changed()

    def _on_mode_change(self, event):
        """Receive layer model mode change event and update checkbox ticks.

        Available modes for labels layer are:
        * PAN_ZOOM
        * TRANSFORM
        * PICK
        * PAINT
        * ERASE
        * FILL
        * POLYGON

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.

        Raises
        ------
        ValueError
            Raise error if event.mode is not PAN_ZOOM, PICK, PAINT, ERASE, FILL,
            TRANSFORM or POLYGON.
        """
        super()._on_mode_change(event)

    def change_color(self):
        """Change colormap of the label layer."""
        self.layer.new_colormap()

    def _on_editable_or_visible_change(self):
        super()._on_editable_or_visible_change()
        self._set_polygon_tool_state()

    def _on_ndisplay_changed(self):
        self._on_editable_or_visible_change()
        self._set_polygon_tool_state()
        super()._on_ndisplay_changed()

    def _set_polygon_tool_state(self):
        if hasattr(self, 'polygon_button'):
            set_widgets_enabled_with_opacity(
                self, [self.polygon_button], self._is_polygon_tool_enabled()
            )

    def _is_polygon_tool_enabled(self):
        return (
            self.layer.editable
            and self.layer.visible
            and self.layer.n_edit_dimensions == 2
            and self.ndisplay == 2
        )
