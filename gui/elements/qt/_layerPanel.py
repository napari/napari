from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFrame
from ._layerList import QtLayerList
from os.path import dirname, join, realpath
import weakref

dir_path = dirname(realpath(__file__))
path_delete = join(dir_path,'icons','delete.png')

class QtLayerPanel(QWidget):
    def __init__(self, layers):
        super().__init__()

        layout = QVBoxLayout()
        self.layersList = QtLayerList(layers)
        self.layersControls = QtLayerControls(layers)
        layout.addWidget(self.layersControls)
        layout.addWidget(self.layersList)
        self.setLayout(layout)

class QDeleteButton(QPushButton):
    def __init__(self, layers):
        super().__init__()

        self.layers = weakref.proxy(layers)

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
        self.delete_layers()

    def delete_layers(self):
        to_delete = []
        for i in range(len(self.layers)):
            if self.layers[i].selected:
                to_delete.append(i)
        to_delete.reverse()
        for i in to_delete:
            self.layers.pop(i)

    def dragEnterEvent(self, event):
        event.accept()

    def dropEvent(self, event):
        event.accept()
        layerWidget = event.source()
        # if not layerWidget.layer.selected:
        #      self.layers.remove(layerWidget.layer)
        # else:
        #     self.delete_layers()
        print('Dropped!!!')

class QtLayerControls(QFrame):
    def __init__(self, layers):
        super().__init__()

        layout = QHBoxLayout()
        pb = QDeleteButton(layers)
        layout.addWidget(pb)
        layout.addStretch(0)

        self.setLayout(layout)
