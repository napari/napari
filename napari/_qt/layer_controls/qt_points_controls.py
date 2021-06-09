import numpy as np
from qtpy.QtCore import Qt, Slot
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSlider,
)

from ...layers.points._points_constants import SYMBOL_TRANSLATION, Mode
from ...utils.action_manager import action_manager
from ...utils.events import disconnect_events
from ...utils.translations import trans
from ..utils import disable_with_opacity, qt_signals_blocked
from ..widgets.qt_color_swatch import QColorSwatchEdit
from ..widgets.qt_mode_buttons import QtModePushButton, QtModeRadioButton
from .qt_layer_controls_base import QtLayerControls


class QtPointsControls(QtLayerControls):
    """Qt view and controls for the napari Points layer.

    Parameters
    ----------
    layer : napari.layers.Points
        An instance of a napari Points layer.

    Attributes
    ----------
    addition_button : qtpy.QtWidgets.QtModeRadioButton
        Button to add points to layer.
    button_group : qtpy.QtWidgets.QButtonGroup
        Button group of points layer modes (ADD, PAN_ZOOM, SELECT).
    delete_button : qtpy.QtWidgets.QtModePushButton
        Button to delete points from layer.
    edgeColorSwatch : qtpy.QtWidgets.QFrame
        Color swatch showing shapes edge display color.
    edgeComboBox : qtpy.QtWidgets.QComboBox
        Dropdown widget to select display color for shape edges.
    faceColorSwatch : qtpy.QtWidgets.QFrame
        Color swatch showing shapes face display color.
    faceComboBox : qtpy.QtWidgets.QComboBox
        Dropdown widget to select display color for shape faces.
    grid_layout : qtpy.QtWidgets.QGridLayout
        Layout of Qt widget controls for the layer.
    layer : napari.layers.Points
        An instance of a napari Points layer.
    ndimCheckBox : qtpy.QtWidgets.QCheckBox
        Checkbox to indicate whether layer is n-dimensional.
    panzoom_button : qtpy.QtWidgets.QtModeRadioButton
        Button for pan/zoom mode.
    select_button : qtpy.QtWidgets.QtModeRadioButton
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

    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.mode.connect(self._on_mode_change)
        self.layer.events.n_dimensional.connect(self._on_n_dimensional_change)
        self.layer.events.symbol.connect(self._on_symbol_change)
        self.layer.events.size.connect(self._on_size_change)
        self.layer.events.current_edge_color.connect(
            self._on_current_edge_color_change
        )
        self.layer._edge.events.current_color.connect(
            self._on_current_edge_color_change
        )
        self.layer.events.current_face_color.connect(
            self._on_current_face_color_change
        )
        self.layer._face.events.current_color.connect(
            self._on_current_face_color_change
        )
        self.layer.events.editable.connect(self._on_editable_change)
        self.layer.text.events.visible.connect(self._on_text_visibility_change)

        sld = QSlider(Qt.Horizontal)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setMinimum(1)
        sld.setMaximum(100)
        sld.setSingleStep(1)
        value = self.layer.current_size
        sld.setValue(int(value))
        sld.valueChanged.connect(self.changeSize)
        self.sizeSlider = sld

        self.faceColorEdit = QColorSwatchEdit(
            initial_color=self.layer.current_face_color,
            tooltip=trans._('click to set current face color'),
        )
        self.edgeColorEdit = QColorSwatchEdit(
            initial_color=self.layer.current_edge_color,
            tooltip=trans._('click to set current edge color'),
        )
        self.faceColorEdit.color_changed.connect(self.changeFaceColor)
        self.edgeColorEdit.color_changed.connect(self.changeEdgeColor)

        symbol_comboBox = QComboBox()
        current_index = 0
        for index, (data, text) in enumerate(SYMBOL_TRANSLATION.items()):
            data = data.value
            symbol_comboBox.addItem(text, data)

            if data == self.layer.symbol:
                current_index = index

        symbol_comboBox.setCurrentIndex(current_index)
        symbol_comboBox.activated[str].connect(self.changeSymbol)
        self.symbolComboBox = symbol_comboBox

        ndim_cb = QCheckBox()
        ndim_cb.setToolTip(trans._('N-dimensional points'))
        ndim_cb.setChecked(self.layer.n_dimensional)
        ndim_cb.stateChanged.connect(self.change_ndim)
        self.ndimCheckBox = ndim_cb

        self.select_button = QtModeRadioButton(
            layer,
            'select_points',
            Mode.SELECT,
        )
        action_manager.bind_button(
            'napari:activate_points_select_mode', self.select_button
        )
        self.addition_button = QtModeRadioButton(layer, 'add_points', Mode.ADD)
        action_manager.bind_button(
            'napari:activate_points_add_mode', self.addition_button
        )
        self.panzoom_button = QtModeRadioButton(
            layer,
            'pan_zoom',
            Mode.PAN_ZOOM,
            checked=True,
        )
        action_manager.bind_button(
            'napari:activate_points_pan_zoom_mode', self.panzoom_button
        )
        self.delete_button = QtModePushButton(
            layer,
            'delete_shape',
        )
        action_manager.bind_button(
            'napari:delete_selected_points', self.delete_button
        )

        text_disp_cb = QCheckBox()
        text_disp_cb.setToolTip(trans._('toggle text visibility'))
        text_disp_cb.setChecked(self.layer.text.visible)
        text_disp_cb.stateChanged.connect(self.change_text_visibility)
        self.textDispCheckBox = text_disp_cb

        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.select_button)
        self.button_group.addButton(self.addition_button)
        self.button_group.addButton(self.panzoom_button)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.delete_button)
        button_row.addWidget(self.addition_button)
        button_row.addWidget(self.select_button)
        button_row.addWidget(self.panzoom_button)
        button_row.setContentsMargins(0, 0, 0, 5)
        button_row.setSpacing(4)

        # grid_layout created in QtLayerControls
        # addWidget(widget, row, column, [row_span, column_span])
        self.grid_layout.addLayout(button_row, 0, 1)
        self.grid_layout.addWidget(QLabel(trans._('opacity:')), 1, 0)
        self.grid_layout.addWidget(self.opacitySlider, 1, 1)
        self.grid_layout.addWidget(QLabel(trans._('point size:')), 2, 0)
        self.grid_layout.addWidget(self.sizeSlider, 2, 1)
        self.grid_layout.addWidget(QLabel(trans._('blending:')), 3, 0)
        self.grid_layout.addWidget(self.blendComboBox, 3, 1)
        self.grid_layout.addWidget(QLabel(trans._('symbol:')), 4, 0)
        self.grid_layout.addWidget(self.symbolComboBox, 4, 1)
        self.grid_layout.addWidget(QLabel(trans._('face color:')), 5, 0)
        self.grid_layout.addWidget(self.faceColorEdit, 5, 1)
        self.grid_layout.addWidget(QLabel(trans._('edge color:')), 6, 0)
        self.grid_layout.addWidget(self.edgeColorEdit, 6, 1)
        self.grid_layout.addWidget(QLabel(trans._('display text:')), 7, 0)
        self.grid_layout.addWidget(self.textDispCheckBox, 7, 1)
        self.grid_layout.addWidget(QLabel(trans._('n-dim:')), 8, 0)
        self.grid_layout.addWidget(self.ndimCheckBox, 8, 1)
        self.grid_layout.setRowStretch(9, 1)
        self.grid_layout.setColumnStretch(1, 1)
        self.grid_layout.setSpacing(4)

    def _on_mode_change(self, event):
        """Update ticks in checkbox widgets when points layer mode is changed.

        Available modes for points layer are:
        * ADD
        * SELECT
        * PAN_ZOOM

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.

        Raises
        ------
        ValueError
            Raise error if event.mode is not ADD, PAN_ZOOM, or SELECT.
        """
        mode = event.mode
        if mode == Mode.ADD:
            self.addition_button.setChecked(True)
        elif mode == Mode.SELECT:
            self.select_button.setChecked(True)
        elif mode == Mode.PAN_ZOOM:
            self.panzoom_button.setChecked(True)
        else:
            raise ValueError(trans._("Mode not recognized"))

    def changeSymbol(self, text):
        """Change marker symbol of the points on the layer model.

        Parameters
        ----------
        text : int
            Index of current marker symbol of points, eg: '+', '.', etc.
        """
        self.layer.symbol = self.symbolComboBox.currentData()

    def changeSize(self, value):
        """Change size of points on the layer model.

        Parameters
        ----------
        value : float
            Size of points.
        """
        self.layer.current_size = value

    def change_ndim(self, state):
        """Toggle n-dimensional state of label layer.

        Parameters
        ----------
        state : QCheckBox
            Checkbox indicating if label layer is n-dimensional.
        """
        if state == Qt.Checked:
            self.layer.n_dimensional = True
        else:
            self.layer.n_dimensional = False

    def change_text_visibility(self, state):
        """Toggle the visibiltiy of the text.

        Parameters
        ----------
        state : QCheckBox
            Checkbox indicating if text is visible.
        """
        if state == Qt.Checked:
            self.layer.text.visible = True
        else:
            self.layer.text.visible = False

    def _on_text_visibility_change(self, event):
        """Receive layer model text visibiltiy change change event and update checkbox.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        with self.layer.text.events.visible.blocker():
            self.textDispCheckBox.setChecked(self.layer.text.visible)

    def _on_n_dimensional_change(self, event):
        """Receive layer model n-dimensional change event and update checkbox.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        with self.layer.events.n_dimensional.blocker():
            self.ndimCheckBox.setChecked(self.layer.n_dimensional)

    def _on_symbol_change(self, event):
        """Receive marker symbol change event and update the dropdown menu.

        Parameters
        ----------
        event : napari.utils.event.Event
            The napari event that triggered this method.
        """
        with self.layer.events.symbol.blocker():
            self.symbolComboBox.setCurrentIndex(
                self.symbolComboBox.findData(self.layer.symbol)
            )

    def _on_size_change(self, event=None):
        """Receive layer model size change event and update point size slider.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method.
        """
        with self.layer.events.size.blocker():
            value = self.layer.current_size
            self.sizeSlider.setValue(int(value))

    @Slot(np.ndarray)
    def changeFaceColor(self, color: np.ndarray):
        """Update face color of layer model from color picker user input."""
        with self.layer.events.current_face_color.blocker():
            self.layer.current_face_color = color

    @Slot(np.ndarray)
    def changeEdgeColor(self, color: np.ndarray):
        """Update edge color of layer model from color picker user input."""
        with self.layer.events.current_edge_color.blocker():
            self.layer.current_edge_color = color

    def _on_current_face_color_change(self, event=None):
        """Receive layer.current_face_color() change event and update view."""
        with qt_signals_blocked(self.faceColorEdit):
            self.faceColorEdit.setColor(self.layer.current_face_color)

    def _on_current_edge_color_change(self, event=None):
        """Receive layer.current_edge_color() change event and update view."""
        with qt_signals_blocked(self.edgeColorEdit):
            self.edgeColorEdit.setColor(self.layer.current_edge_color)

    def _on_editable_change(self, event=None):
        """Receive layer model editable change event & enable/disable buttons.

        Parameters
        ----------
        event : napari.utils.event.Event, optional
            The napari event that triggered this method, by default None.
        """
        disable_with_opacity(
            self,
            ['select_button', 'addition_button', 'delete_button'],
            self.layer.editable,
        )

    def close(self):
        """Disconnect events when widget is closing."""
        disconnect_events(self.layer.text.events, self)
        super().close()
