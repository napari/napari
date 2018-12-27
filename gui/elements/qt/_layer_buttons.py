from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFrame, QCheckBox
from os.path import join
from ...resources.icons import icons_dir

path_delete = join(icons_dir,'delete.png')
path_add = join(icons_dir,'add.png')
path_off = join(icons_dir,'annotation_off.png')
path_on = join(icons_dir,'annotation_on.png')

class QtLayerButtons(QFrame):
    def __init__(self, layers):
        super().__init__()

        self.layers = layers
        self.annotationCheckBox = QtAnnotationCheckBox(self.layers)
        self.deleteButton = QtDeleteButton(self.layers)
        self.addButton = QtAddButton(self.layers)

        layout = QHBoxLayout()
        layout.addWidget(self.annotationCheckBox)
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
        styleSheet = """QPushButton {background-color:lightGray; border-radius: 3px;}
            QPushButton:pressed {background-color:rgb(0, 153, 255); border-radius: 3px;}
            QPushButton:hover {background-color:rgb(0, 153, 255); border-radius: 3px;}"""
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
        styleSheet = """QPushButton {background-color:lightGray; border-radius: 3px;}
            QPushButton:pressed {background-color:rgb(0, 153, 255); border-radius: 3px;}
            QPushButton:hover {background-color:rgb(0, 153, 255); border-radius: 3px;}"""
        self.setStyleSheet(styleSheet)
        self.clicked.connect(self.layers.viewer._new_markers)

class QtAnnotationCheckBox(QCheckBox):
    def __init__(self, layers):
        super().__init__()

        self.layers = layers
        self.setToolTip('Annotation mode')
        self.setChecked(False)
        styleSheet = """QCheckBox {background-color:lightGray; border-radius: 3px;}
                        QCheckBox::indicator {subcontrol-position: center center; subcontrol-origin: content;
                            width: 28px; height: 28px;}
                        QCheckBox::indicator:checked {background-color:rgb(0, 153, 255); border-radius: 3px;
                            image: url(""" + path_off + """);}
                        QCheckBox::indicator:unchecked {image: url(""" + path_off + """);}
                        QCheckBox::indicator:unchecked:hover {image: url(""" + path_on + ");}"
        self.setStyleSheet(styleSheet)
        self.stateChanged.connect(lambda state=self: self.layers.viewer._set_annotation(state))
        self.layers.viewer.events.annotation.connect(self._set_annotation)

    def _set_annotation(self, event):
        with self.layers.viewer.events.blocker(self._set_annotation):
            self.setChecked(self.layers.viewer.annotation)
