from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QFrame, QCheckBox)
from os.path import join
from ....resources import resources_dir

path_delete = join(resources_dir, 'icons', 'delete.png')
path_add = join(resources_dir, 'icons', 'add.png')

styleSheet = """QPushButton {background-color:lightGray; border-radius: 3px;}
                QPushButton:pressed {background-color:rgb(0, 153, 255);
                border-radius: 3px;}
                QPushButton:hover {background-color:rgb(0, 153, 255);
                border-radius: 3px;}"""


class QtLayersButtons(QFrame):
    def __init__(self, layers):
        super().__init__()

        self.layers = layers
        self.deleteButton = QtDeleteButton(self.layers)
        self.addButton = QtAddButton(self.layers)

        layout = QHBoxLayout()
        layout.addStretch(0)
        layout.addWidget(self.addButton)
        layout.addWidget(self.deleteButton)
        self.setLayout(layout)


class QtDeleteButton(QPushButton):
    def __init__(self, layers):
        super().__init__()

        self.layers = layers
        self.setIcon(QIcon(path_delete))
        self.setFixedWidth(28)
        self.setFixedHeight(28)
        self.setToolTip('Delete layers')
        self.setAcceptDrops(True)
        self.setStyleSheet(styleSheet)
        self.clicked.connect(self.layers.remove_selected)

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


class QtAddButton(QPushButton):
    def __init__(self, layers):
        super().__init__()

        self.layers = layers
        self.setIcon(QIcon(path_add))
        self.setFixedWidth(28)
        self.setFixedHeight(28)
        self.setToolTip('Add layer')
        self.setStyleSheet(styleSheet)
        self.clicked.connect(self.layers.viewer._new_markers)
