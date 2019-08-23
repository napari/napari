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
)

from ...layers import Points
from ...layers.points._constants import Mode, Symbol
from .qt_base_layer import QtLayerControls, QtLayerProperties


class QtPointsControls(QtLayerControls, layer=Points):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.mode.connect(self.set_mode)

        self.select_button = QtSelectButton(layer)
        self.addition_button = QtAdditionButton(layer)
        self.panzoom_button = QtPanZoomButton(layer)
        self.delete_button = QtDeletePointsButton(layer)

        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.select_button)
        self.button_group.addButton(self.addition_button)
        self.button_group.addButton(self.panzoom_button)

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 20, 10, 10)
        layout.addWidget(self.panzoom_button)
        layout.addWidget(self.select_button)
        layout.addWidget(self.addition_button)
        layout.addWidget(self.delete_button)
        layout.addStretch(0)
        self.setLayout(layout)
        self.setMouseTracking(True)

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


class QtPanZoomButton(QRadioButton):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        self.setToolTip('Pan/zoom mode')
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
        self.setToolTip('Select mode')
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
        self.setToolTip('Addition mode')
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
        self.setToolTip('Delete selected')
        self.clicked.connect(self.layer.remove_selected)


class QtPointsProperties(QtLayerProperties, layer=Points):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.n_dimensional.connect(self._on_n_dim_change)
        self.layer.events.symbol.connect(self._on_symbol_change)
        self.layer.events.size.connect(self._on_size_change)
        self.layer.events.edge_color.connect(self._on_edge_color_change)
        self.layer.events.face_color.connect(self._on_face_color_change)

        sld = QSlider(Qt.Horizontal, self)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setFixedWidth(110)
        sld.setMinimum(1)
        sld.setMaximum(100)
        sld.setSingleStep(1)
        value = self.layer.size
        sld.setValue(int(value))
        sld.valueChanged[int].connect(lambda value=sld: self.changeSize(value))
        self.sizeSlider = sld
        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(QLabel('size:'), row, self.name_column)
        self.grid_layout.addWidget(sld, row, self.property_column)

        face_comboBox = QComboBox()
        colors = self.layer._colors
        for c in colors:
            face_comboBox.addItem(c)
        index = face_comboBox.findText(
            self.layer.face_color, Qt.MatchFixedString
        )
        face_comboBox.setCurrentIndex(index)
        face_comboBox.activated[str].connect(
            lambda text=face_comboBox: self.changeFaceColor(text)
        )
        self.faceComboBox = face_comboBox
        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(
            QLabel('face_color:'), row, self.name_column
        )
        self.grid_layout.addWidget(face_comboBox, row, self.property_column)

        edge_comboBox = QComboBox()
        colors = self.layer._colors
        for c in colors:
            edge_comboBox.addItem(c)
        index = edge_comboBox.findText(
            self.layer.edge_color, Qt.MatchFixedString
        )
        edge_comboBox.setCurrentIndex(index)
        edge_comboBox.activated[str].connect(
            lambda text=edge_comboBox: self.changeEdgeColor(text)
        )
        self.edgeComboBox = edge_comboBox
        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(
            QLabel('edge_color:'), row, self.name_column
        )
        self.grid_layout.addWidget(edge_comboBox, row, self.property_column)

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
        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(QLabel('symbol:'), row, self.name_column)
        self.grid_layout.addWidget(symbol_comboBox, row, self.property_column)

        ndim_cb = QCheckBox()
        ndim_cb.setToolTip('N-dimensional points')
        ndim_cb.setChecked(self.layer.n_dimensional)
        ndim_cb.stateChanged.connect(
            lambda state=ndim_cb: self.change_ndim(state)
        )
        self.ndimCheckBox = ndim_cb
        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(QLabel('n-dim:'), row, self.name_column)
        self.grid_layout.addWidget(ndim_cb, row, self.property_column)

        self.setExpanded(False)

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

    def _on_face_color_change(self, event):
        with self.layer.events.face_color.blocker():
            index = self.faceComboBox.findText(
                self.layer.face_color, Qt.MatchFixedString
            )
            self.faceComboBox.setCurrentIndex(index)
