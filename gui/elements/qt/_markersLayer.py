from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QComboBox, QSlider
from collections import Iterable
import numpy as np
from ._layer import QtLayer

class QtMarkersLayer(QtLayer):
    def __init__(self, layer):
        super().__init__(layer)

        self.grid_layout.addWidget(QLabel('size:'), 2, 0)
        sld = QSlider(Qt.Horizontal, self)
        sld.setFocusPolicy(Qt.NoFocus)
        #sld.setInvertedAppearance(True)
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
        self.grid_layout.addWidget(sld, 2, 1)

        face_comboBox = QComboBox()
        colors = self.layer._colors
        for c in colors:
           face_comboBox.addItem(c)
        index = face_comboBox.findText(self.layer.face_color, Qt.MatchFixedString)
        if index >= 0:
           face_comboBox.setCurrentIndex(index)
        face_comboBox.activated[str].connect(lambda text=face_comboBox: self.changeFaceColor(text))
        self.grid_layout.addWidget(QLabel('face_color:'), 3, 0)
        self.grid_layout.addWidget(face_comboBox, 3, 1)

        edge_comboBox = QComboBox()
        colors = self.layer._colors
        for c in colors:
           edge_comboBox.addItem(c)
        index = edge_comboBox.findText(self.layer.edge_color, Qt.MatchFixedString)
        if index >= 0:
           edge_comboBox.setCurrentIndex(index)
        edge_comboBox.activated[str].connect(lambda text=edge_comboBox: self.changeEdgeColor(text))
        self.grid_layout.addWidget(QLabel('edge_color:'), 4, 0)
        self.grid_layout.addWidget(edge_comboBox, 4, 1)

        symbol_comboBox = QComboBox()
        symbols = self.layer._marker_types
        for s in symbols:
           symbol_comboBox.addItem(s)
        index = symbol_comboBox.findText(self.layer.symbol, Qt.MatchFixedString)
        if index >= 0:
           symbol_comboBox.setCurrentIndex(index)
        symbol_comboBox.activated[str].connect(lambda text=symbol_comboBox: self.changeSymbol(text))
        self.grid_layout.addWidget(QLabel('symbol:'), 5, 0)
        self.grid_layout.addWidget(symbol_comboBox, 5, 1)

        blend_comboBox = QComboBox()
        for blend in self.layer._blending_modes:
            blend_comboBox.addItem(blend)
        index = blend_comboBox.findText(self.layer._blending, Qt.MatchFixedString)
        if index >= 0:
            blend_comboBox.setCurrentIndex(index)
        blend_comboBox.activated[str].connect(lambda text=blend_comboBox: self.changeBlending(text))
        self.grid_layout.addWidget(QLabel('blending:'), 6, 0)
        self.grid_layout.addWidget(blend_comboBox, 6, 1)

        self.setExpanded(False)

    def changeFaceColor(self, text):
        self.layer.face_color = text

    def changeEdgeColor(self, text):
        self.layer.edge_color = text

    def changeSymbol(self, text):
        self.layer.symbol = text

    def changeSize(self, value):
        self.layer.size = value

    def changeBlending(self, text):
        self.layer.blending = text
