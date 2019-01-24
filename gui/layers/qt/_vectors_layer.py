#!/usr/bin/env python
# title           : this_python_file.py
# description     :This will create a header for a python script.
# author          :bryant.chhun
# date            :1/16/19
# version         :0.0
# usage           :python this_python_file.py -flags
# notes           :
# python_version  :3.6

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QLabel, QComboBox, QSpinBox
from ._base_layer import QtLayer

class QtVectorsLayer(QtLayer):

    # compute_average = pyqtSignal(object)

    def __init__(self, layer):
        super().__init__(layer)

        #averaging combobox
        averaging_combobox = QComboBox()
        avg_dims = self.layer._avg_dims
        for avg in avg_dims:
            averaging_combobox.addItem(avg)
        index = averaging_combobox.findText(self.layer.averaging, Qt.MatchFixedString)
        if index >= 0:
            averaging_combobox.setCurrentIndex(index)
            averaging_combobox.activated[str].connect(lambda text=averaging_combobox: self.changeAvgType(text))

        self.grid_layout.addWidget(QLabel('averaging:'), 3, 0)
        self.grid_layout.addWidget(averaging_combobox, 3, 1)

        # line width in pixels
        width_field = QSpinBox()
        value = self.layer.width
        width_field.setValue(value)
        width_field.valueChanged.connect(self.changeWidth)
        self.grid_layout.addWidget(QLabel('size:'), 4, 0)
        self.grid_layout.addWidget(width_field, 4, 1)

        # vector color adjustment and widget
        face_comboBox = QComboBox()
        colors = self.layer._colors
        for c in colors:
           face_comboBox.addItem(c)
        index = face_comboBox.findText(self.layer.color, Qt.MatchFixedString)
        if index >= 0:
           face_comboBox.setCurrentIndex(index)
        face_comboBox.activated[str].connect(lambda text=face_comboBox: self.changeFaceColor(text))
        self.grid_layout.addWidget(QLabel('face_color:'), 5, 0)
        self.grid_layout.addWidget(face_comboBox, 5, 1)

        # line connector type.  Only two built in: Segments or Connected
        # connector_comboBox = QComboBox()
        # connector_type = self.layer._connector_types
        # for c in connector_type:
        #     connector_comboBox.addItem(c)
        # index = connector_comboBox.findText(self.layer.connector, Qt.MatchFixedString)
        # if index >= 0:
        #     connector_comboBox.setCurrentIndex(index)
        # connector_comboBox.activated[str].connect(lambda text=connector_comboBox: self.changeConnectorType(text))
        # self.grid_layout.addWidget(QLabel('Connector:'), 6, 0)
        # self.grid_layout.addWidget(connector_comboBox, 6, 1)

        self.setExpanded(False)

    def changeFaceColor(self, text):
        self.layer.color = text

    def changeConnectorType(self, text):
        self.layer.connector = text

    def changeAvgType(self, text):
        self.layer.averaging = text
        # self.compute_average.emit(text)

    def changeWidth(self, value):
        self.layer.width = value
