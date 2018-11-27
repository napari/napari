from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFrame, QCheckBox
from ._layerList import QtLayerList
from os.path import dirname, join, realpath
from numpy import empty

dir_path = dirname(realpath(__file__))
path_delete = join(dir_path,'icons','delete.png')
path_add = join(dir_path,'icons','add.png')
path_off = join(dir_path,'icons','annotation_off.png')
path_on = join(dir_path,'icons','annotation_on.png')

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

class QAddLayerButton(QPushButton):
    def __init__(self, layers):
        super().__init__()

        self.layers = layers
        self.setIcon(QIcon(path_add))
        self.setFixedWidth(28)
        self.setFixedHeight(28)
        self.setToolTip('Add layer')
        self.clicked.connect(self.on_click)
        styleSheet = """QPushButton {background-color:lightGray; border-radius: 3px;}
            QPushButton:pressed {background-color:rgb(71,143,205); border-radius: 3px;}
            QPushButton:hover {background-color:rgb(71,143,205); border-radius: 3px;}"""
        self.setStyleSheet(styleSheet)

    def on_click(self):
        self.layers.viewer.add_markers(empty((0, self.layers.viewer.max_dims)))

class QAnnotationCheckBox(QCheckBox):
    def __init__(self, layers):
        super().__init__()

        self.layers = layers
        #self.setFixedWidth(28)
        #self.setFixedHeight(28)
        self.setToolTip('Annotation mode')
        self.setChecked(False)
        self.stateChanged.connect(lambda state=self: self.changeAnnotation(state))
        styleSheet = """QCheckBox {background-color:lightGray; border-radius: 3px;}
                        QCheckBox::indicator {subcontrol-position: center center; subcontrol-origin: content;
                            width: 28px; height: 28px;}
                        QCheckBox::indicator:unchecked:hover {background-color:rgb(71,143,205); border-radius: 3px;}
                        QCheckBox::indicator:unchecked {image: url(""" + path_off + """);}
                        QCheckBox::indicator:checked {image: url(""" + path_on + ");}"
        self.setStyleSheet(styleSheet)

    def changeAnnotation(self, state):
        if state == Qt.Checked:
            self.layers.viewer.annotation = True
        else:
            self.layers.viewer.annotation = False

class QtLayerControls(QFrame):
    def __init__(self, layers):
        super().__init__()

        layout = QHBoxLayout()
        layout.addWidget(QAnnotationCheckBox(layers))
        layout.addStretch(0)
        layout.addWidget(QAddLayerButton(layers))
        layout.addWidget(QDeleteButton(layers))

        self.setLayout(layout)
