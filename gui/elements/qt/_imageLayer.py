from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QComboBox
from ._layer import QtLayer

class QtImageLayer(QtLayer):
    def __init__(self, layer):
        super().__init__(layer)

        self.expanded_height = 110
        comboBox = QComboBox()
        for cmap in self.layer.colormaps:
            comboBox.addItem(cmap)
        index = comboBox.findText('hot', Qt.MatchFixedString)
        if index >= 0:
            comboBox.setCurrentIndex(index)
        comboBox.activated[str].connect(lambda text=comboBox: self.changeColor(text))

        self.grid_layout.addWidget(QLabel('colormap:'), 2, 0)
        self.grid_layout.addWidget(comboBox, 2, 1)
        self.setExpanded(False)

    def changeColor(self, text):
        self.layer.colormap = text
