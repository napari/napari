from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFrame, QCheckBox
from os.path import dirname, join, realpath

dir_path = dirname(realpath(__file__))
path_delete = join(dir_path, '..', '..', 'icons','delete.png')
path_add = join(dir_path, '..', '..', 'icons','add.png')
path_off = join(dir_path,'..', '..', 'icons','annotation_off.png')
path_on = join(dir_path,'..', '..', 'icons','annotation_on.png')

class QtLayerButtons(QFrame):
    def __init__(self):
        super().__init__()

        self.annotationCheckBox = QAnnotationCheckBox()
        self.deleteButton = QDeleteButton()
        self.addLayerButton = QAddLayerButton()

        layout = QHBoxLayout()
        layout.addWidget(self.annotationCheckBox)
        layout.addStretch(0)
        layout.addWidget(self.addLayerButton)
        layout.addWidget(self.deleteButton)
        self.setLayout(layout)

class QDeleteButton(QPushButton):
    def __init__(self):
        super().__init__()

        self.setIcon(QIcon(path_delete))
        self.setFixedWidth(28)
        self.setFixedHeight(28)
        self.setToolTip('Delete layers')
        self.setAcceptDrops(True)
        styleSheet = """QPushButton {background-color:lightGray; border-radius: 3px;}
            QPushButton:pressed {background-color:rgb(0, 153, 255); border-radius: 3px;}
            QPushButton:hover {background-color:rgb(0, 153, 255); border-radius: 3px;}"""
        self.setStyleSheet(styleSheet)

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
    def __init__(self):
        super().__init__()

        self.setIcon(QIcon(path_add))
        self.setFixedWidth(28)
        self.setFixedHeight(28)
        self.setToolTip('Add layer')
        styleSheet = """QPushButton {background-color:lightGray; border-radius: 3px;}
            QPushButton:pressed {background-color:rgb(0, 153, 255); border-radius: 3px;}
            QPushButton:hover {background-color:rgb(0, 153, 255); border-radius: 3px;}"""
        self.setStyleSheet(styleSheet)

class QAnnotationCheckBox(QCheckBox):
    def __init__(self):
        super().__init__()

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
