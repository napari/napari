
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QComboBox, QSpinBox

from ..._base_layer import QtLayer


class QtVectorsLayer(QtLayer):

    def __init__(self, layer):
        super().__init__(layer)

        # vector color adjustment and widget
        face_comboBox = QComboBox()
        colors = self.layer._colors
        for c in colors:
            face_comboBox.addItem(c)
        index = face_comboBox.findText(self.layer.color, Qt.MatchFixedString)
        if index >= 0:
            face_comboBox.setCurrentIndex(index)
        face_comboBox.activated[str].connect(
            lambda text=face_comboBox: self.changeFaceColor(text))
        self.grid_layout.addWidget(QLabel('color:'), 3, 0)
        self.grid_layout.addWidget(face_comboBox, 3, 1)

        # line width in pixels
        width_field = QSpinBox()
        value = self.layer.width
        width_field.setValue(value)
        width_field.valueChanged.connect(self.changeWidth)
        self.grid_layout.addWidget(QLabel('width:'), 4, 0)
        self.grid_layout.addWidget(width_field, 4, 1)

        # averaging combobox
        averaging_combobox = QComboBox()
        avg_dims = self.layer._avg_dims
        for avg in avg_dims:
            averaging_combobox.addItem(avg)
        index = averaging_combobox.findText(
            self.layer.averaging, Qt.MatchFixedString)
        if index >= 0:
            averaging_combobox.setCurrentIndex(index)
            averaging_combobox.activated[str].connect(
                lambda text=averaging_combobox: self.changeAvgType(text))

        self.grid_layout.addWidget(QLabel('averaging:'), 5, 0)
        self.grid_layout.addWidget(averaging_combobox, 5, 1)

        # line length
        length_field = QSpinBox()
        value = self.layer.length
        length_field.setValue(value)
        length_field.valueChanged.connect(self.changeLength)
        self.grid_layout.addWidget(QLabel('length:'), 6, 0)
        self.grid_layout.addWidget(length_field, 6, 1)

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
        self.layer.length = int(value)
