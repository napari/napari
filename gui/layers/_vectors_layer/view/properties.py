#!/usr/bin/env python
# title           : properties.py
# description     :qt vectors layer
# author          :bryant.chhun
# date            :1/16/19
# version         :0.0
# usage           :
# notes           :
# python_version  :3.6

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QComboBox, QSpinBox

from ..._base_layer import QtLayer


class QtVectorsLayer(QtLayer):

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
        self.grid_layout.addWidget(QLabel('width:'), 4, 0)
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
        self.grid_layout.addWidget(QLabel('color:'), 5, 0)
        self.grid_layout.addWidget(face_comboBox, 5, 1)
        
        # length slider bar
        # sld = QSlider(Qt.Horizontal, self)
        # sld.setFocusPolicy(Qt.NoFocus)
        # sld.setFixedWidth(75)
        # sld.setMinimum(0)
        # sld.setMaximum(10)
        # sld.setSingleStep(1)
        # value = self.layer.length
        # if isinstance(value, Iterable):
        #     if isinstance(value, list):
        #         value = np.asarray(value)
        #     value = value.mean()
        # sld.setValue(int(value))
        # sld.valueChanged[int].connect(lambda value=sld: self.changeLength(value))
        # self.grid_layout.addWidget(QLabel('length:'), 6, 0)
        # self.grid_layout.addWidget(sld, 6, 1)
        
        length_field = QSpinBox()
        value = self.layer.length
        length_field.setValue(value)
        length_field.valueChanged.connect(self.changeLength)
        self.grid_layout.addWidget(QLabel('length:'), 6, 0)
        self.grid_layout.addWidget(length_field, 6, 1)
        
        # length_combobox = QComboBox()
        # for length in ['1','5','10','15','20','25','30']:
        #     length_combobox.addItem(length)
        # # index = length_combobox.findText(self.layer.length, Qt.MatchFixedString)
        # index = length_combobox.findData(self.layer.length)
        # print(index)
        # if index >= 0:
        #     length_combobox.setCurrentIndex(index)
        #     length_combobox.activated[str].connect(lambda text=length_combobox: self.changeLength(value))
        #
        # self.grid_layout.addWidget(QLabel('length:'), 6, 0)
        # self.grid_layout.addWidget(length_combobox, 6, 1)

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

    def changeWidth(self, value):
        self.layer.width = value
    
    def changeLength(self, value):
        print('length changed to = '+str(value))
        self.layer.length = int(value)