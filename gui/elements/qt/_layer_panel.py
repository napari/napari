from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from ._layer_list import QtLayerList
from ._layer_buttons import QtLayerButtons


class QtLayerPanel(QWidget):
    def __init__(self, layers):
        super().__init__()

        self.layers = layers
        self.layerList = QtLayerList(self.layers)
        self.layerButtons = QtLayerButtons(self.layers)
        layout = QVBoxLayout()
        layout.addWidget(self.layerList)
        layout.addWidget(self.layerButtons)
        self.setLayout(layout)
        self.setMinimumSize(QSize(250, 250))
