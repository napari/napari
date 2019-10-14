from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QLabel,
    QComboBox,
    QSlider,
    QCheckBox,
    QButtonGroup,
    QVBoxLayout,
    QRadioButton,
    QPushButton,
    QFrame,
)
from vispy.color import Color
from .qt_base_layer import QtLayerControls
from ...layers.points._constants import Mode, Symbol


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
        value = self.layer.size
        sld.setValue(int(value))
        sld.valueChanged[int].connect(lambda value=sld: self.changeSize(value))
        self.sizeSlider = sld

        face_comboBox = QComboBox()
        colors = self.layer._colors
        for c in colors:
            face_comboBox.addItem(c)
        face_comboBox.activated[str].connect(
            lambda text=face_comboBox: self.changeFaceColor(text)
        )
        self.faceComboBox = face_comboBox
        self.faceColorSwatch = QFrame()
        self.faceColorSwatch.setObjectName('swatch')
        self.faceColorSwatch.setToolTip('Face color swatch')
        self._on_face_color_change(None)

        edge_comboBox = QComboBox()
        colors = self.layer._colors
        for c in colors:
            edge_comboBox.addItem(c)
        edge_comboBox.activated[str].connect(
            lambda text=edge_comboBox: self.changeEdgeColor(text)
        )
        self.edgeComboBox = edge_comboBox
        self.edgeColorSwatch = QFrame()
        self.edgeColorSwatch.setObjectName('swatch')
        self.edgeColorSwatch.setToolTip('Edge color swatch')
        self._on_edge_color_change(None)

        symbol_comboBox = QComboBox()
        for s in Symbol:
            symbol_comboBox.addItem(str(s))
        index = symbol_comboBox.findText(
            self.layer.symbol, Qt.MatchFixedString
        )
        symbol_comboBox.setCurrentIndex(index)
        symbol_comboBox.activated[str].connect(
            lambda text=symbol_comboBox: self.changeSymbol(text)
        )
        self.symbolComboBox = symbol_comboBox

        ndim_cb = QCheckBox()
        ndim_cb.setToolTip('N-dimensional points')
        ndim_cb.setChecked(self.layer.n_dimensional)
        ndim_cb.stateChanged.connect(
            lambda state=ndim_cb: self.change_ndim(state)
        )
        self.ndimCheckBox = ndim_cb

        self.select_button = QtSelectButton(layer)
        self.addition_button = QtAdditionButton(layer)
        self.panzoom_button = QtPanZoomButton(layer)
        self.delete_button = QtDeletePointsButton(layer)

        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.select_button)
        self.button_group.addButton(self.addition_button)
        self.button_group.addButton(self.panzoom_button)

        # grid_layout created in QtLayerControls
        # addWidget(widget, row, column, [row_span, column_span])
        self.grid_layout.addWidget(self.panzoom_button, 0, 6)
        self.grid_layout.addWidget(self.select_button, 0, 5)
        self.grid_layout.addWidget(self.addition_button, 0, 4)
        self.grid_layout.addWidget(self.delete_button, 0, 3)
        self.grid_layout.addWidget(QLabel('opacity:'), 1, 0, 1, 3)
        self.grid_layout.addWidget(self.opacitySilder, 1, 3, 1, 4)
        self.grid_layout.addWidget(QLabel('point size:'), 2, 0, 1, 3)
        self.grid_layout.addWidget(self.sizeSlider, 2, 3, 1, 4)
        self.grid_layout.addWidget(QLabel('blending:'), 3, 0, 1, 3)
        self.grid_layout.addWidget(self.blendComboBox, 3, 3, 1, 4)
        self.grid_layout.addWidget(QLabel('symbol:'), 4, 0, 1, 3)
        self.grid_layout.addWidget(self.symbolComboBox, 4, 3, 1, 4)
        self.grid_layout.addWidget(QLabel('face color:'), 5, 0, 1, 3)
        self.grid_layout.addWidget(self.faceComboBox, 5, 3, 1, 3)
        self.grid_layout.addWidget(self.faceColorSwatch, 5, 6)
        self.grid_layout.addWidget(QLabel('edge color:'), 6, 0, 1, 3)
        self.grid_layout.addWidget(self.edgeComboBox, 6, 3, 1, 3)
        self.grid_layout.addWidget(self.edgeColorSwatch, 6, 6)
        self.grid_layout.addWidget(QLabel('n-dim:'), 7, 0, 1, 3)
        self.grid_layout.addWidget(self.ndimCheckBox, 7, 3)
        self.grid_layout.setRowStretch(8, 1)
        self.grid_layout.setVerticalSpacing(4)

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
            raise ValueError("Mode not recongnized")

    def changeFaceColor(self, text):
        self.layer.face_color = text

    def changeEdgeColor(self, text):
        self.layer.edge_color = text

    def changeSymbol(self, text):
        self.layer.symbol = text

    def changeSize(self, value):
        self.layer.size = value

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

    def _on_size_change(self, event):
        with self.layer.events.size.blocker():
            value = self.layer.size
            self.sizeSlider.setValue(int(value))

    def _on_edge_color_change(self, event):
        with self.layer.events.edge_color.blocker():
            index = self.edgeComboBox.findText(
                self.layer.edge_color, Qt.MatchFixedString
            )
            self.edgeComboBox.setCurrentIndex(index)
        color = Color(self.layer.edge_color).hex
        self.edgeColorSwatch.setStyleSheet("background-color: " + color)

    def _on_face_color_change(self, event):
        with self.layer.events.face_color.blocker():
            index = self.faceComboBox.findText(
                self.layer.face_color, Qt.MatchFixedString
            )
            self.faceComboBox.setCurrentIndex(index)
        color = Color(self.layer.face_color).hex
        self.faceColorSwatch.setStyleSheet("background-color: " + color)

    def _on_editable_change(self, event):
        self.select_button.setEnabled(self.layer.editable)
        self.addition_button.setEnabled(self.layer.editable)
        self.delete_button.setEnabled(self.layer.editable)


class QtPanZoomButton(QRadioButton):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        self.setToolTip('Pan/zoom')
        self.setChecked(True)
        self.toggled.connect(lambda state=self: self._set_mode(state))
        self.setFixedWidth(28)

    def _set_mode(self, bool):
        with self.layer.events.mode.blocker():
            if bool:
                self.layer.mode = Mode.PAN_ZOOM


class QtSelectButton(QRadioButton):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        self.setToolTip('Select points')
        self.setChecked(False)
        self.toggled.connect(lambda state=self: self._set_mode(state))
        self.setFixedWidth(28)

    def _set_mode(self, bool):
        with self.layer.events.mode.blocker():
            if bool:
                self.layer.mode = Mode.SELECT


class QtAdditionButton(QRadioButton):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        self.setToolTip('Add points')
        self.setChecked(False)
        self.toggled.connect(lambda state=self: self._set_mode(state))
        self.setFixedWidth(28)

    def _set_mode(self, bool):
        with self.layer.events.mode.blocker():
            if bool:
                self.layer.mode = Mode.ADD


class QtDeletePointsButton(QPushButton):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        self.setFixedWidth(28)
        self.setFixedHeight(28)
        self.setToolTip('Delete selected points')
        self.clicked.connect(self.layer.remove_selected)
