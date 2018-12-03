from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QComboBox
from ._layer import QtLayer

class QtImageLayer(QtLayer):
    def __init__(self, layer):
        super().__init__(layer)

        comboBox = QComboBox()
        for cmap in self.layer.colormaps:
            comboBox.addItem(cmap)
        index = comboBox.findText('hot', Qt.MatchFixedString)
        if index >= 0:
            comboBox.setCurrentIndex(index)
        comboBox.activated[str].connect(lambda text=comboBox: self.changeColor(text))
        self.grid_layout.addWidget(QLabel('colormap:'), 2, 0)
        self.grid_layout.addWidget(comboBox, 2, 1)

        interp_comboBox = QComboBox()
        for interp in self.layer._interpolation_names:
            interp_comboBox.addItem(interp)
        index = interp_comboBox.findText(self.layer.interpolation, Qt.MatchFixedString)
        if index >= 0:
            interp_comboBox.setCurrentIndex(index)
        interp_comboBox.activated[str].connect(lambda text=interp_comboBox: self.changeInterpolation(text))
        self.grid_layout.addWidget(QLabel('interpolation:'), 3, 0)
        self.grid_layout.addWidget(interp_comboBox, 3, 1)

        blend_comboBox = QComboBox()
        for blend in self.layer._blending_modes:
            blend_comboBox.addItem(blend)
        index = blend_comboBox.findText(self.layer._blending, Qt.MatchFixedString)
        if index >= 0:
            blend_comboBox.setCurrentIndex(index)
        blend_comboBox.activated[str].connect(lambda text=blend_comboBox: self.changeBlending(text))
        self.grid_layout.addWidget(QLabel('blending:'), 4, 0)
        self.grid_layout.addWidget(blend_comboBox, 4, 1)

        self.setExpanded(False)

    def changeColor(self, text):
        self.layer.colormap = text

    def changeInterpolation(self, text):
        self.layer.interpolation = text

    def changeBlending(self, text):
        self.layer.blending = text
