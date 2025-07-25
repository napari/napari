import contextlib
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt, Slot
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
)
from superqt import QLabeledSlider

from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari._qt.utils import qt_signals_blocked
from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit
from napari._qt.widgets.qt_mode_buttons import QtModePushButton
from napari.layers.points._points_constants import (
    SYMBOL_TRANSLATION,
    SYMBOL_TRANSLATION_INVERTED,
    Mode,
)
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
    layer : napari.layers.Points
        An instance of a napari Points layer.
    MODE : Enum
        Available modes in the associated layer.
    PAN_ZOOM_ACTION_NAME : str
        String id for the pan-zoom action to bind to the pan_zoom button.
    TRANSFORM_ACTION_NAME : str
        String id for the transform action to bind to the transform button.
    addition_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to add points to layer.
    button_group : qtpy.QtWidgets.QButtonGroup
        Button group of points layer modes (ADD, PAN_ZOOM, SELECT).
    delete_button : qtpy.QtWidgets.QtModePushButton
        Button to delete points from layer.
    borderColorEdit : QColorSwatchEdit
        Widget to select display color for points borders.
    faceColorEdit : QColorSwatchEdit
        Widget to select display color for points faces.
    outOfSliceCheckBox : qtpy.QtWidgets.QCheckBox
        Checkbox to indicate whether to render out of slice.
    panzoom_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to activate move camera mode for layer.
    transform_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select transform mode.
    select_button : napari._qt.widgets.qt_mode_button.QtModeRadioButton
        Button to select points from layer.
    sizeSlider : qtpy.QtWidgets.QSlider
        Slider controlling size of points.
    symbolComboBox : qtpy.QtWidgets.QComboBox
        Drop down list of symbol options for points markers.

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

        self.layer.events.out_of_slice_display.connect(
            self._on_out_of_slice_display_change
        )
        self.layer.events.symbol.connect(self._on_current_symbol_change)
        self.layer.events.size.connect(self._on_current_size_change)
        self.layer.events.current_size.connect(self._on_current_size_change)
        self.layer.events.current_border_color.connect(
            self._on_current_border_color_change
        )
        self.layer._border.events.current_color.connect(
            self._on_current_border_color_change
        )
        self.layer.events.current_face_color.connect(
            self._on_current_face_color_change
        )
        self.layer._face.events.current_color.connect(
            self._on_current_face_color_change
        )
        self.layer.events.current_symbol.connect(
            self._on_current_symbol_change
        )
        self.layer.text.events.visible.connect(self._on_text_visibility_change)

        sld = QLabeledSlider(Qt.Orientation.Horizontal)
        sld.setToolTip(
            trans._(
                'Change the size of currently selected points and any added afterwards.'
            )
        )
        sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sld.setMinimum(1)
        if self.layer.size.size:
            max_value = max(100, int(np.max(self.layer.size)) + 1)
        else:
            max_value = 100
        sld.setMaximum(max_value)
        sld.setSingleStep(1)
        value = self.layer.current_size
        sld.setValue(int(value))
        sld.valueChanged.connect(self.changeCurrentSize)
        self.sizeSlider = sld

        self.faceColorEdit = QColorSwatchEdit(
            initial_color=self.layer.current_face_color,
            tooltip=trans._(
                'Click to set the face color of currently selected points and any added afterwards.'
            ),
        )
        self.borderColorEdit = QColorSwatchEdit(
            initial_color=self.layer.current_border_color,
            tooltip=trans._(
                'Click to set the border color of currently selected points and any added afterwards.'
            ),
        )
        self.faceColorEdit.color_changed.connect(self.changeCurrentFaceColor)
        self.borderColorEdit.color_changed.connect(
            self.changeCurrentBorderColor
        )

        sym_cb = QComboBox()
        sym_cb.setToolTip(
            trans._(
                'Change the symbol of currently selected points and any added afterwards.'
            )
        )
        current_index = 0
        for index, (symbol_string, text) in enumerate(
            SYMBOL_TRANSLATION.items()
        ):
            symbol_string = symbol_string.value
            sym_cb.addItem(text, symbol_string)

            if symbol_string == self.layer.current_symbol:
                current_index = index

        sym_cb.setCurrentIndex(current_index)
        sym_cb.currentTextChanged.connect(self.changeCurrentSymbol)
        self.symbolComboBox = sym_cb

        self.outOfSliceCheckBox = QCheckBox()
        self.outOfSliceCheckBox.setToolTip(trans._('Out of slice display'))
        self.outOfSliceCheckBox.setChecked(self.layer.out_of_slice_display)
        self.outOfSliceCheckBox.stateChanged.connect(self.change_out_of_slice)

        self.textDispCheckBox = QCheckBox()
        self.textDispCheckBox.setToolTip(trans._('Toggle text visibility'))
        self.textDispCheckBox.setChecked(self.layer.text.visible)
        self.textDispCheckBox.stateChanged.connect(self.change_text_visibility)

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

        self.layout().addRow(self.button_grid)
        self.layout().addRow(self.opacityLabel, self.opacitySlider)
        self.layout().addRow(trans._('blending:'), self.blendComboBox)
        self.layout().addRow(
            trans._('projection mode:'), self.projectionComboBox
        )
        self.layout().addRow(trans._('point size:'), self.sizeSlider)
        self.layout().addRow(trans._('symbol:'), self.symbolComboBox)
        self.layout().addRow(trans._('face color:'), self.faceColorEdit)
        self.layout().addRow(trans._('border color:'), self.borderColorEdit)
        self.layout().addRow(trans._('display text:'), self.textDispCheckBox)
        self.layout().addRow(trans._('out of slice:'), self.outOfSliceCheckBox)

    def changeCurrentSymbol(self, text):
        """Change marker symbol of the points on the layer model.

        Parameters
        ----------
        text : int
            Index of current marker symbol of points, eg: '+', '.', etc.
        """
        with self.layer.events.symbol.blocker(self._on_current_symbol_change):
            self.layer.current_symbol = SYMBOL_TRANSLATION_INVERTED[text]

    def changeCurrentSize(self, value):
        """Change size of points on the layer model.

        Parameters
        ----------
        value : float
            Size of points.
        """
        with self.layer.events.current_size.blocker(
            self._on_current_size_change
        ):
            self.layer.current_size = value

    def change_out_of_slice(self, state):
        """Toggleout of slice display of points layer.

        Parameters
        ----------
        state : QCheckBox
            Checkbox indicating whether to render out of slice.
        """
        # needs cast to bool for Qt6
        with self.layer.events.out_of_slice_display.blocker(
            self._on_out_of_slice_display_change
        ):
            self.layer.out_of_slice_display = bool(state)

    def change_text_visibility(self, state):
        """Toggle the visibility of the text.

        Parameters
        ----------
        state : QCheckBox
            Checkbox indicating if text is visible.
        """
        with self.layer.text.events.visible.blocker(
            self._on_text_visibility_change
        ):
            # needs cast to bool for Qt6
            self.layer.text.visible = bool(state)

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

    def _on_text_visibility_change(self):
        """Receive layer model text visibiltiy change event and update checkbox."""
        with qt_signals_blocked(self.textDispCheckBox):
            self.textDispCheckBox.setChecked(self.layer.text.visible)

    def _on_out_of_slice_display_change(self):
        """Receive layer model out_of_slice_display change event and update checkbox."""
        with qt_signals_blocked(self.outOfSliceCheckBox):
            self.outOfSliceCheckBox.setChecked(self.layer.out_of_slice_display)

    def _on_current_symbol_change(self):
        """Receive marker symbol change event and update the dropdown menu."""
        with qt_signals_blocked(self.symbolComboBox):
            self.symbolComboBox.setCurrentIndex(
                self.symbolComboBox.findData(self.layer.current_symbol.value)
            )

    def _on_current_size_change(self):
        """Receive layer model size change event and update point size slider."""
        with qt_signals_blocked(self.sizeSlider):
            value = self.layer.current_size
            min_val = min(value) if isinstance(value, list) else value
            max_val = max(value) if isinstance(value, list) else value
            if min_val < self.sizeSlider.minimum():
                self.sizeSlider.setMinimum(max(1, int(min_val - 1)))
            if max_val > self.sizeSlider.maximum():
                self.sizeSlider.setMaximum(int(max_val + 1))
            with contextlib.suppress(TypeError):
                self.sizeSlider.setValue(int(value))

    @Slot(np.ndarray)
    def changeCurrentFaceColor(self, color: np.ndarray):
        """Update face color of layer model from color picker user input."""
        with self.layer.events.current_face_color.blocker(
            self._on_current_face_color_change
        ):
            self.layer.current_face_color = color

    @Slot(np.ndarray)
    def changeCurrentBorderColor(self, color: np.ndarray):
        """Update border color of layer model from color picker user input."""
        with self.layer.events.current_border_color.blocker(
            self._on_current_border_color_change
        ):
            self.layer.current_border_color = color

    def _on_current_face_color_change(self):
        """Receive layer.current_face_color() change event and update view."""
        with qt_signals_blocked(self.faceColorEdit):
            self.faceColorEdit.setColor(self.layer.current_face_color)

    def _on_current_border_color_change(self):
        """Receive layer.current_border_color() change event and update view."""
        with qt_signals_blocked(self.borderColorEdit):
            self.borderColorEdit.setColor(self.layer.current_border_color)

    def _on_ndisplay_changed(self):
        self.layer.editable = not (self.layer.ndim == 2 and self.ndisplay == 3)
        super()._on_ndisplay_changed()

    def close(self):
        """Disconnect events when widget is closing."""
        disconnect_events(self.layer.text.events, self)
        super().close()
