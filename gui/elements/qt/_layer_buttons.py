from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFrame, QCheckBox
from os.path import dirname, join, realpath
from numpy import empty

dir_path = dirname(realpath(__file__))
path_delete = join(dir_path, '..', '..', 'icons','delete.png')
path_add = join(dir_path, '..', '..', 'icons','add.png')
path_off = join(dir_path,'..', '..', 'icons','annotation_off.png')
path_on = join(dir_path,'..', '..', 'icons','annotation_on.png')

class QtLayerButtons(QFrame):
    def __init__(self, layers):
        super().__init__()

        self.annotationCheckBox = QAnnotationCheckBox(layers)
        layout = QHBoxLayout()
        layout.addWidget(self.annotationCheckBox)
        layout.addStretch(0)
        layout.addWidget(QAddLayerButton(layers))
        layout.addWidget(QDeleteButton(layers))

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
            QPushButton:pressed {background-color:rgb(0, 153, 255); border-radius: 3px;}
            QPushButton:hover {background-color:rgb(0, 153, 255); border-radius: 3px;}"""
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
            QPushButton:pressed {background-color:rgb(0, 153, 255); border-radius: 3px;}
            QPushButton:hover {background-color:rgb(0, 153, 255); border-radius: 3px;}"""
        self.setStyleSheet(styleSheet)

    def on_click(self):
        if self.layers.viewer.dimensions.max_dims == 0:
            empty_markers = empty((0, 2))
        else:
            empty_markers = empty((0, self.layers.viewer.dimensions.max_dims))
        self.layers.viewer.add_markers(empty_markers)

class QAnnotationCheckBox(QCheckBox):
    def __init__(self, layers):
        super().__init__()

        self.layers = layers
        self.setToolTip('Annotation mode')
        self.setChecked(False)
        self.stateChanged.connect(lambda state=self: self.changeAnnotation(state))
        styleSheet = """QCheckBox {background-color:lightGray; border-radius: 3px;}
                        QCheckBox::indicator {subcontrol-position: center center; subcontrol-origin: content;
                            width: 28px; height: 28px;}
                        QCheckBox::indicator:checked {background-color:rgb(0, 153, 255); border-radius: 3px;
                            image: url(""" + path_off + """);}
                        QCheckBox::indicator:unchecked {image: url(""" + path_off + """);}
                        QCheckBox::indicator:unchecked:hover {image: url(""" + path_on + ");}"
        self.setStyleSheet(styleSheet)

    def changeAnnotation(self, state):
        if state == Qt.Checked:
            self.layers.viewer._set_annotation_mode(True)
        else:
            self.layers.viewer._set_annotation_mode(False)
