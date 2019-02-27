from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from .list import QtLayersList
from .buttons import QtLayersButtons


class QtLayersPanel(QWidget):
    def __init__(self, layers):
        super().__init__()

        self.layers = layers
        self.layersList = QtLayersList(self.layers)
        self.layersButtons = QtLayersButtons(self.layers)
        layout = QVBoxLayout()
        layout.addWidget(self.layersList)
        layout.addWidget(self.layersButtons)
        self.setLayout(layout)
        self.setMinimumSize(QSize(250, 250))
