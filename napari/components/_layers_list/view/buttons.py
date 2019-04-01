from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QFrame, QCheckBox)

from .... import resources


class QtLayersButtons(QFrame):
    def __init__(self, layers):
        super().__init__()

        self.layers = layers
        self.deleteButton = QtDeleteButton(self.layers)
        self.newMarkersButton = QtNewMarkersButton(self.layers)
        self.newShapesButton = QtNewShapesButton(self.layers)
        self.newLabelsButton = QtNewLabelsButton(self.layers)

        layout = QHBoxLayout()
        layout.addStretch(0)
        layout.addWidget(self.newMarkersButton)
        layout.addWidget(self.newShapesButton)
        layout.addWidget(self.newLabelsButton)
        layout.addWidget(self.deleteButton)
        self.setLayout(layout)


class QtDeleteButton(QPushButton):
    def __init__(self, layers):
        super().__init__()

        self.layers = layers
        self.setIcon(QIcon(':/icons/delete.png'))
        self.setFixedWidth(28)
        self.setFixedHeight(28)
        self.setToolTip('Delete layers')
        self.setAcceptDrops(True)
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


class QtNewMarkersButton(QPushButton):
    def __init__(self, layers):
        super().__init__()

        self.layers = layers
        self.setIcon(QIcon(':icons/new_markers.png'))
        self.setFixedWidth(28)
        self.setFixedHeight(28)
        self.setToolTip('New markers layer')
        self.clicked.connect(self.layers.viewer._new_markers)


class QtNewShapesButton(QPushButton):
    def __init__(self, layers):
        super().__init__()

        self.layers = layers
        self.setIcon(QIcon(':icons/new_shapes.png'))
        self.setFixedWidth(28)
        self.setFixedHeight(28)
        self.setToolTip('New shapes layer')
        self.clicked.connect(self.layers.viewer._new_shapes)


class QtNewLabelsButton(QPushButton):
    def __init__(self, layers):
        super().__init__()

        self.layers = layers
        self.setIcon(QIcon(':icons/new_labels.png'))
        self.setFixedWidth(28)
        self.setFixedHeight(28)
        self.setToolTip('New labels layer')
        self.clicked.connect(self.layers.viewer._new_labels)
