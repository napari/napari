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

from ...layers.points._constants import Mode, Symbol
from ..qt_color_dialog import QColorSwatchEdit
from ..qt_mode_buttons import QtModePushButton, QtModeRadioButton
from ..utils import qt_signals_blocked
from .qt_base_layer import QtLayerControls


class QtPointsControls(QtLayerControls):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.mode.connect(self.set_mode)
        self.layer.events.n_dimensional.connect(self._on_n_dim_change)
        self.layer.events.symbol.connect(self._on_symbol_change)
        self.layer.events.size.connect(self._on_size_change)
        self.layer.events.current_edge_color.connect(
            self._on_edge_color_change
        )
        self.layer.events.current_face_color.connect(
            self._on_face_color_change
        )
        self.layer.events.editable.connect(self._on_editable_change)

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
            tooltip='click to set current face color',
        )
        self.edgeColorEdit = QColorSwatchEdit(
            initial_color=self.layer.current_edge_color,
            tooltip='click to set current edge color',
        )
        self.faceColorEdit.color_changed.connect(self.changeFaceColor)
        self.edgeColorEdit.color_changed.connect(self.changeEdgeColor)

        symbol_comboBox = QComboBox()
        symbol_comboBox.addItems([str(s) for s in Symbol])
        index = symbol_comboBox.findText(
            self.layer.symbol, Qt.MatchFixedString
        )
        symbol_comboBox.setCurrentIndex(index)
        symbol_comboBox.activated[str].connect(self.changeSymbol)
        self.symbolComboBox = symbol_comboBox

        ndim_cb = QCheckBox()
        ndim_cb.setToolTip('N-dimensional points')
        ndim_cb.setChecked(self.layer.n_dimensional)
        ndim_cb.stateChanged.connect(self.change_ndim)
        self.ndimCheckBox = ndim_cb

        self.select_button = QtModeRadioButton(
            layer, 'select_points', Mode.SELECT, tooltip='Select points'
        )
        self.addition_button = QtModeRadioButton(
            layer, 'add_points', Mode.ADD, tooltip='Add points'
        )
        self.panzoom_button = QtModeRadioButton(
            layer, 'pan_zoom', Mode.PAN_ZOOM, tooltip='Pan/zoom', checked=True
        )
        self.delete_button = QtModePushButton(
            layer,
            'delete_shape',
            slot=self.layer.remove_selected,
            tooltip='Delete selected points',
        )

        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.select_button)
        self.button_group.addButton(self.addition_button)
        self.button_group.addButton(self.panzoom_button)

        button_row = QHBoxLayout()
        button_row.addWidget(self.delete_button)
        button_row.addWidget(self.addition_button)
        button_row.addWidget(self.select_button)
        button_row.addWidget(self.panzoom_button)
        button_row.addStretch(1)
        button_row.setSpacing(4)

        # grid_layout created in QtLayerControls
        # addWidget(widget, row, column, [row_span, column_span])
        self.grid_layout.addLayout(button_row, 0, 1, 1, 2)
        self.grid_layout.addWidget(QLabel('opacity:'), 1, 0)
        self.grid_layout.addWidget(self.opacitySlider, 1, 1, 1, 2)
        self.grid_layout.addWidget(QLabel('point size:'), 2, 0)
        self.grid_layout.addWidget(self.sizeSlider, 2, 1, 1, 2)
        self.grid_layout.addWidget(QLabel('blending:'), 3, 0)
        self.grid_layout.addWidget(self.blendComboBox, 3, 1, 1, 2)
        self.grid_layout.addWidget(QLabel('symbol:'), 4, 0)
        self.grid_layout.addWidget(self.symbolComboBox, 4, 1, 1, 2)
        self.grid_layout.addWidget(QLabel('face color:'), 5, 0)
        self.grid_layout.addWidget(self.faceColorEdit, 5, 1, 1, 2)
        self.grid_layout.addWidget(QLabel('edge color:'), 6, 0)
        self.grid_layout.addWidget(self.edgeColorEdit, 6, 1, 1, 2)
        self.grid_layout.addWidget(QLabel('n-dim:'), 7, 0)
        self.grid_layout.addWidget(self.ndimCheckBox, 7, 1)
        self.grid_layout.setRowStretch(8, 1)
        self.grid_layout.setColumnStretch(1, 1)
        self.grid_layout.setSpacing(4)

    def mouseMoveEvent(self, event):
        self.layer.status = self.layer.mode

    def set_mode(self, event):
        mode = event.mode
        if mode == Mode.ADD:
            self.addition_button.setChecked(True)
        elif mode == Mode.SELECT:
            self.select_button.setChecked(True)
        elif mode == Mode.PAN_ZOOM:
            self.panzoom_button.setChecked(True)
        else:
            raise ValueError("Mode not recognized")

    def changeSymbol(self, text):
        self.layer.symbol = text

    def changeSize(self, value):
        self.layer.current_size = value

    def change_ndim(self, state):
        if state == Qt.Checked:
            self.layer.n_dimensional = True
        else:
            self.layer.n_dimensional = False

    def _on_n_dim_change(self, event):
        with self.layer.events.n_dimensional.blocker():
            self.ndimCheckBox.setChecked(self.layer.n_dimensional)

    def _on_symbol_change(self, event):
        with self.layer.events.symbol.blocker():
            index = self.symbolComboBox.findText(
                self.layer.symbol, Qt.MatchFixedString
            )
            self.symbolComboBox.setCurrentIndex(index)

    def _on_size_change(self, event=None):
        with self.layer.events.size.blocker():
            value = self.layer.current_size
            self.sizeSlider.setValue(int(value))

    @Slot(np.ndarray)
    def changeFaceColor(self, color: np.ndarray):
        """Update the layer model based on user input in the color picker."""
        with self.layer.events.current_face_color.blocker():
            self.layer.current_face_color = color

    @Slot(np.ndarray)
    def changeEdgeColor(self, color: np.ndarray):
        """Update the layer model based on user input in the color picker."""
        with self.layer.events.current_edge_color.blocker():
            self.layer.current_edge_color = color

    def _on_face_color_change(self, event=None):
        """Receive layer.current_face_color() change event and update view."""
        with qt_signals_blocked(self.faceColorEdit):
            self.faceColorEdit.setColor(self.layer.current_face_color)

    def _on_edge_color_change(self, event=None):
        """Receive layer.current_edge_color() change event and update view."""
        with qt_signals_blocked(self.edgeColorEdit):
            self.edgeColorEdit.setColor(self.layer.current_edge_color)

    def _on_editable_change(self, event=None):
        self.select_button.setEnabled(self.layer.editable)
        self.addition_button.setEnabled(self.layer.editable)
        self.delete_button.setEnabled(self.layer.editable)
