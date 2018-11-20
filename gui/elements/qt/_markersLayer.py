from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QComboBox
from ._layer import QtLayer

class QtMarkersLayer(QtLayer):
    def __init__(self, layer):
        super().__init__(layer)

        comboBox = QComboBox()
        colors = ['red', 'blue', 'green', 'white']
        for c in colors:
           comboBox.addItem(c)
        index = comboBox.findText('white', Qt.MatchFixedString)
        if index >= 0:
           comboBox.setCurrentIndex(index)
        comboBox.activated[str].connect(lambda text=comboBox: self.changeColor(text))

        self.grid_layout.addWidget(QLabel('face_color:'), 2, 0)
        self.grid_layout.addWidget(comboBox, 2, 1)
        self.setExpanded(False)

    def changeColor(self, text):
        self.layer.face_color = text
