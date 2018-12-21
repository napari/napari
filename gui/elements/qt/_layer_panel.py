from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from ._layer_list import QtLayerList
from ._layer_buttons import QtLayerButtons


class QtLayerPanel(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        self.layerList = QtLayerList()
        self.layerButtons = QtLayerButtons()
        layout.addWidget(self.layerList)
        layout.addWidget(self.layerButtons)
        self.setLayout(layout)
        self.setMinimumSize(QSize(250, 250))
