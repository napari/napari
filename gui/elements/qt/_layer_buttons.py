from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QFrame, QCheckBox)
from os.path import join
from ...resources import resources_dir

path_delete = join(resources_dir, 'icons', 'delete.png')
path_new = join(resources_dir, 'icons', 'new.png')
path_add_off = join(resources_dir, 'icons', 'add_off.png')
path_add_on = join(resources_dir, 'icons', 'add_on.png')
path_select_off = join(resources_dir, 'icons', 'select_off.png')
path_select_on = join(resources_dir, 'icons', 'select_on.png')

styleSheet = """QPushButton {background-color:lightGray; border-radius: 3px;}
                QPushButton:pressed {background-color:rgb(0, 153, 255);
                border-radius: 3px;}
                QPushButton:hover {background-color:rgb(0, 153, 255);
                border-radius: 3px;}"""


class QtLayerButtons(QFrame):
    def __init__(self, layers):
        super().__init__()

        self.layers = layers
        self.selectCheckBox = QtSelectCheckBox(self.layers)
        self.additionCheckBox = QtAdditionCheckBox(self.layers)
        self.deleteButton = QtDeleteButton(self.layers)
        self.newLayerButton = QtNewLayerButton(self.layers)

        layout = QHBoxLayout()
        layout.addWidget(self.selectCheckBox)
        layout.addWidget(self.additionCheckBox)
        layout.addStretch(0)
        layout.addWidget(self.newLayerButton)
        layout.addWidget(self.deleteButton)
        self.setLayout(layout)

        self.layers.viewer.events.mode.connect(self._set_button)

    def _set_button(self, event):
        with self.layers.viewer.events.blocker(self._set_button):
            if self.layers.viewer.mode == 'add':
                self.selectCheckBox.setChecked(False)
                self.additionCheckBox.setChecked(True)
            elif self.layers.viewer.mode == 'select':
                self.additionCheckBox.setChecked(False)
                self.selectCheckBox.setChecked(True)
            else:
                self.selectCheckBox.setChecked(False)
                self.additionCheckBox.setChecked(False)

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


class QtNewLayerButton(QPushButton):
    def __init__(self, layers):
        super().__init__()

        self.layers = layers
        self.setIcon(QIcon(path_new))
        self.setFixedWidth(28)
        self.setFixedHeight(28)
        self.setToolTip('Add layer')
        self.setStyleSheet(styleSheet)
        self.clicked.connect(self.layers.viewer._new_markers)


class QtSelectCheckBox(QCheckBox):
    def __init__(self, layers):
        super().__init__()

        self.layers = layers
        self.setToolTip('Select mode')
        self.setChecked(False)
        styleSheet = """QCheckBox {background-color:lightGray;
                        border-radius: 3px;}
                        QCheckBox::indicator {subcontrol-position:
                        center center; subcontrol-origin: content;
                        width: 28px; height: 28px;}
                        QCheckBox::indicator:checked {
                        background-color:rgb(0, 153, 255); border-radius: 3px;
                        image: url(""" + path_select_off + """);}
                        QCheckBox::indicator:unchecked
                        {image: url(""" + path_select_off + """);}
                        QCheckBox::indicator:unchecked:hover
                        {image: url(""" + path_select_on + ");}"
        self.setStyleSheet(styleSheet)
        self.stateChanged.connect(lambda state=self: self._set_mode(state))

    def _set_mode(self, bool):
        with self.layers.viewer.events.blocker(self._set_mode):
            if bool:
                self.layers.viewer._set_mode('select')
            else:
                self.layers.viewer._set_mode(None)


class QtAdditionCheckBox(QCheckBox):
    def __init__(self, layers):
        super().__init__()

        self.layers = layers
        self.setToolTip('Addition mode')
        self.setChecked(False)
        styleSheet = """QCheckBox {background-color:lightGray;
                        border-radius: 3px;}
                        QCheckBox::indicator {subcontrol-position:
                        center center; subcontrol-origin: content;
                        width: 28px; height: 28px;}
                        QCheckBox::indicator:checked {
                        background-color:rgb(0, 153, 255); border-radius: 3px;
                        image: url(""" + path_add_off + """);}
                        QCheckBox::indicator:unchecked
                        {image: url(""" + path_add_off + """);}
                        QCheckBox::indicator:unchecked:hover
                        {image: url(""" + path_add_on + ");}"
        self.setStyleSheet(styleSheet)
        self.stateChanged.connect(lambda state=self: self._set_mode(state))

    def _set_mode(self, bool):
        with self.layers.viewer.events.blocker(self._set_mode):
            if bool:
                self.layers.viewer._set_mode('add')
            else:
                self.layers.viewer._set_mode(None)
