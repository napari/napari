from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFrame
from ._layerList import QtLayerList
from os.path import dirname, join, realpath

dir_path = dirname(realpath(__file__))
path_delete = join(dir_path,'icons','delete.png')

class QtLayerPanel(QWidget):
    def __init__(self, layers):
        super().__init__()

        layout = QVBoxLayout()
        self.layersList = QtLayerList(layers)
        self.layersControls = QtLayerControls(layers)
        layout.addWidget(self.layersList)
        layout.addWidget(self.layersControls)
        self.setLayout(layout)

class QDeleteButton(QPushButton):
    def __init__(self, layers):
        super().__init__()

        self.layers = layers
        self.setIcon(QIcon(path_delete))
        self.setFixedWidth(28)
        self.setFixedHeight(28)
        self.setToolTip('Delete layers')
        self.clicked.connect(self.on_click)
        self.setAcceptDrops(True)
        styleSheet = """QPushButton {background-color:lightGray; border-radius: 3px;}
            QPushButton:pressed {background-color:rgb(71,143,205); border-radius: 3px;}
            QPushButton:hover {background-color:rgb(71,143,205); border-radius: 3px;}"""
        self.setStyleSheet(styleSheet)

    def on_click(self):
        self.layers.remove_selected()

    def dragEnterEvent(self, event):
        event.accept()
        self.hover = True
        self.update()

    def dragLeaveEvent(self, event):
        event.ignore()
        self.hover = False
        self.update()

    def dropEvent(self, event):
        event.setDropAction(Qt.CopyAction)
        event.accept()

class QtLayerControls(QFrame):
    def __init__(self, layers):
        super().__init__()

        layout = QHBoxLayout()
        layout.addStretch(0)
        layout.addWidget(QDeleteButton(layers))

        self.setLayout(layout)
