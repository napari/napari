from PyQt5.QtWidgets import QWidget, QVBoxLayout
from ._layer_list import QtLayerList
from ._layer_buttons import QtLayerButtons


class QtLayerPanel(QWidget):
    def __init__(self, layers):
        super().__init__()

        layout = QVBoxLayout()
        self.layerList = QtLayerList(layers)
        self.layerButtons = QtLayerButtons()
        layout.addWidget(self.layerList)
        layout.addWidget(self.layerButtons)
        self.setLayout(layout)
