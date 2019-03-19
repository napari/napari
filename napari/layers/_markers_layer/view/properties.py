from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QComboBox, QSlider, QCheckBox
from collections import Iterable
import numpy as np
from ..._base_layer import QtLayer


class QtMarkersLayer(QtLayer):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.n_dimensional.connect(self._on_n_dim_change)
        self.layer.events.symbol.connect(self._on_symbol_change)
        self.layer.events.size.connect(self._on_size_change)
        self.layer.events.edge_color.connect(self._on_edge_color_change)
        self.layer.events.face_color.connect(self._on_face_color_change)

        sld = QSlider(Qt.Horizontal, self)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setFixedWidth(75)
        sld.setMinimum(0)
        sld.setMaximum(100)
        sld.setSingleStep(1)
        value = self.layer.size
        if isinstance(value, Iterable):
            if isinstance(value, list):
                value = np.asarray(value)
            value = value.mean()
        sld.setValue(int(value))
        sld.valueChanged[int].connect(lambda value=sld: self.changeSize(value))
        self.sizeSlider = sld
        self.grid_layout.addWidget(QLabel('size:'), 3, 0)
        self.grid_layout.addWidget(sld, 3, 1)

        face_comboBox = QComboBox()
        colors = self.layer._colors
        for c in colors:
            face_comboBox.addItem(c)
        index = face_comboBox.findText(
            self.layer.face_color, Qt.MatchFixedString)
        face_comboBox.setCurrentIndex(index)
        face_comboBox.activated[str].connect(
            lambda text=face_comboBox: self.changeFaceColor(text))
        self.faceComboBox = face_comboBox
        self.grid_layout.addWidget(QLabel('face_color:'), 4, 0)
        self.grid_layout.addWidget(face_comboBox, 4, 1)

        edge_comboBox = QComboBox()
        colors = self.layer._colors
        for c in colors:
            edge_comboBox.addItem(c)
        index = edge_comboBox.findText(
            self.layer.edge_color, Qt.MatchFixedString)
        edge_comboBox.setCurrentIndex(index)
        edge_comboBox.activated[str].connect(
            lambda text=edge_comboBox: self.changeEdgeColor(text))
        self.edgeComboBox = edge_comboBox
        self.grid_layout.addWidget(QLabel('edge_color:'), 5, 0)
        self.grid_layout.addWidget(edge_comboBox, 5, 1)

        symbol_comboBox = QComboBox()
        symbols = self.layer._marker_types
        for s in symbols:
            symbol_comboBox.addItem(s)
        index = symbol_comboBox.findText(
            self.layer.symbol, Qt.MatchFixedString)
        symbol_comboBox.setCurrentIndex(index)
        symbol_comboBox.activated[str].connect(
            lambda text=symbol_comboBox: self.changeSymbol(text))
        self.symbolComboBox = symbol_comboBox
        self.grid_layout.addWidget(QLabel('symbol:'), 6, 0)
        self.grid_layout.addWidget(symbol_comboBox, 6, 1)

        ndim_cb = QCheckBox()
        ndim_cb.setToolTip('N-dimensional markers')
        ndim_cb.setChecked(self.layer.n_dimensional)
        ndim_cb.stateChanged.connect(lambda state=ndim_cb:
                                     self.change_ndim(state))
        self.ndimCheckBox = ndim_cb
        self.grid_layout.addWidget(QLabel('n-dim:'), 7, 0)
        self.grid_layout.addWidget(ndim_cb, 7, 1)

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
                self.layer.symbol, Qt.MatchFixedString)
            self.symbolComboBox.setCurrentIndex(index)

    def _on_size_change(self, event):
        with self.layer.events.size.blocker():
            value = self.layer.size
            if isinstance(value, Iterable):
                if isinstance(value, list):
                    value = np.asarray(value)
                value = value.mean()
            self.sizeSlider.setValue(int(value))

    def _on_edge_color_change(self, event):
        with self.layer.events.edge_color.blocker():
            index = self.edgeComboBox.findText(
                self.layer.edge_color, Qt.MatchFixedString)
            self.edgeComboBox.setCurrentIndex(index)

    def _on_face_color_change(self, event):
        with self.layer.events.face_color.blocker():
            index = self.faceComboBox.findText(
                self.layer.face_color, Qt.MatchFixedString)
            self.faceComboBox.setCurrentIndex(index)
