from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFrame
from ._layerList import QtLayerList

class QtLayerPanel(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        self.layersList = QtLayerList()
        self.layersControls = QtLayerControls()
        layout.addWidget(self.layersControls)
        layout.addWidget(self.layersList)
        self.setLayout(layout)

class QtLayerControls(QFrame):
    def __init__(self):
        super().__init__()

        layout = QHBoxLayout()

        pb = QPushButton('D')
        layout.addWidget(pb)
        self.setLayout(layout)
