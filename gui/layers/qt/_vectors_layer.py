#!/usr/bin/env python
# title           : this_python_file.py
# description     :This will create a header for a python script.
# author          :bryant.chhun
# date            :1/16/19
# version         :0.0
# usage           :python this_python_file.py -flags
# notes           :
# python_version  :3.6

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QComboBox, QSlider
from collections import Iterable
import numpy as np
from ._base_layer import QtLayer

class QtVectorsLayer(QtLayer):
    def __init__(self, layer):
        super().__init__(layer)

        self.grid_layout.addWidget(QLabel('size:'), 3, 0)

        # set layer's width slider attributes and connect signals
        sld = QSlider(Qt.Horizontal, self)
        sld.setFocusPolicy(Qt.NoFocus)
        #sld.setInvertedAppearance(True)
        sld.setFixedWidth(75)
        sld.setMinimum(0)
        sld.setMaximum(100)
        sld.setSingleStep(1)

        value = self.layer.width
        if isinstance(value, Iterable):
            if isinstance(value, list):
                value = np.asarray(value)
            value = value.mean()
        sld.setValue(int(value))
        sld.valueChanged[int].connect(lambda value=sld: self.changeWidth(value))
        self.grid_layout.addWidget(sld, 3, 1)

        # marker face color adjustment and widget
        face_comboBox = QComboBox()
        colors = self.layer._colors
        for c in colors:
           face_comboBox.addItem(c)
        index = face_comboBox.findText(self.layer.color, Qt.MatchFixedString)
        if index >= 0:
           face_comboBox.setCurrentIndex(index)
        face_comboBox.activated[str].connect(lambda text=face_comboBox: self.changeFaceColor(text))
        self.grid_layout.addWidget(QLabel('face_color:'), 4, 0)
        self.grid_layout.addWidget(face_comboBox, 4, 1)

        connector_comboBox = QComboBox()
        connector_type = self.layer._connector_types
        for c in connector_type:
            connector_comboBox.addItem(c)
        index = connector_comboBox.findText(self.layer.connect, Qt.MatchFixedString)
        if index >= 0:
            connector_comboBox.setCurrentIndex(index)
        connector_comboBox.activated[str].connect(lambda text=connector_comboBox: self.changeConnectorType(text))
        self.grid_layout.addWidget(QLabel('Connector:'), 5, 0)
        self.grid_layout.addWidget(connector_comboBox, 5, 1)

        self.setExpanded(False)

    def changeFaceColor(self, text):
        self.layer.face_color = text

    def changeConnectorType(self, text):
        self.layer.connect = text

    # def changeEdgeColor(self, text):
    #     self.layer.edge_color = text
    #
    # def changeSymbol(self, text):
    #     self.layer.symbol = text
    #
    def changeWidth(self, value):
        self.layer.width = value
