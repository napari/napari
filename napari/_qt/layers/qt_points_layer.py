from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QLabel,
    QComboBox,
    QSlider,
    QCheckBox,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
)

from .qt_base_layer import QtLayerControls
from ...layers.points._constants import Mode, Symbol
from ..qt_mode_buttons import QtModeRadioButton, QtModePushButton


class QtPointsControls(QtLayerControls):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.mode.connect(self.set_mode)
        self.layer.events.n_dimensional.connect(self._on_n_dim_change)
        self.layer.events.symbol.connect(self._on_symbol_change)
        self.layer.events.size.connect(self._on_size_change)
        self.layer.events.edge_color.connect(self._on_edge_color_change)
        self.layer.events.face_color.connect(self._on_face_color_change)
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

        face_comboBox = QComboBox()
        face_comboBox.addItems(self.layer._colors)
        face_comboBox.activated[str].connect(self.changeFaceColor)
        self.faceComboBox = face_comboBox
        self.faceColorSwatch = QFrame()
        self.faceColorSwatch.setObjectName('swatch')
        self.faceColorSwatch.setToolTip('Face color swatch')
        self._on_face_color_change()

        edge_comboBox = QComboBox()
        edge_comboBox.addItems(self.layer._colors)
        edge_comboBox.activated[str].connect(self.changeEdgeColor)
        self.edgeComboBox = edge_comboBox
        self.edgeColorSwatch = QFrame()
        self.edgeColorSwatch.setObjectName('swatch')
        self.edgeColorSwatch.setToolTip('Edge color swatch')
        self._on_edge_color_change()

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
        self.grid_layout.addWidget(self.faceComboBox, 5, 2)
        self.grid_layout.addWidget(self.faceColorSwatch, 5, 1)
        self.grid_layout.addWidget(QLabel('edge color:'), 6, 0)
        self.grid_layout.addWidget(self.edgeComboBox, 6, 2)
        self.grid_layout.addWidget(self.edgeColorSwatch, 6, 1)
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

    def changeFaceColor(self, text):
        self.layer.current_face_color = text

    def changeEdgeColor(self, text):
        self.layer.current_edge_color = text

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

    def _on_edge_color_change(self, event=None):
        """Change element's edge color based on user-provided value.

        The new color (read from layer.current_edge_color) is a string -
        either the color's name or its hex representation. This color has
        already been verified by "transform_color". This value has to be
        looked up in the color list of the layer and displayed in the
        combobox. If it's not in the combobox the method will add it and
        then display it, for future use.
        """
        color = self.layer.current_edge_color
        with self.layer.events.edge_color.blocker():
            index = self.edgeComboBox.findText(color, Qt.MatchFixedString)
            if index == -1:
                self.edgeComboBox.addItem(color)
                index = self.edgeComboBox.findText(color, Qt.MatchFixedString)
            self.edgeComboBox.setCurrentIndex(index)
        self.edgeColorSwatch.setStyleSheet(f"background-color: {color}")

    def _on_face_color_change(self, event=None):
        """Change element's face color based user-provided value.

        The new color (read from layer.current_face_color) is a string -
        either the color's name or its hex representation. This color has
        already been verified by "transform_color". This value has to be
        looked up in the color list of the layer and displayed in the
        combobox. If it's not in the combobox the method will add it and
        then display it, for future use.
        """
        color = self.layer.current_face_color
        with self.layer.events.face_color.blocker():
            index = self.faceComboBox.findText(color, Qt.MatchFixedString)
            if index == -1:
                self.faceComboBox.addItem(color)
                index = self.faceComboBox.findText(color, Qt.MatchFixedString)
            self.faceComboBox.setCurrentIndex(index)
        self.faceColorSwatch.setStyleSheet(f"background-color: {color}")

    def _on_editable_change(self, event=None):
        self.select_button.setEnabled(self.layer.editable)
        self.addition_button.setEnabled(self.layer.editable)
        self.delete_button.setEnabled(self.layer.editable)
