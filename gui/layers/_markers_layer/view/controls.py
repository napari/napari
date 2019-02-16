from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QCheckBox, QVBoxLayout, QPushButton, QFrame

from os.path import join
from ....resources import resources_dir

path_add_off = join(resources_dir, 'icons', 'add_off.png')
path_add_on = join(resources_dir, 'icons', 'add_on.png')
path_select_off = join(resources_dir, 'icons', 'select_off.png')
path_select_on = join(resources_dir, 'icons', 'select_on.png')

styleSheet = """QPushButton {background-color:lightGray; border-radius: 3px;}
                QPushButton:pressed {background-color:rgb(0, 153, 255);
                border-radius: 3px;}
                QPushButton:hover {background-color:rgb(0, 153, 255);
                border-radius: 3px;}"""

    # self._cursors = {
    #         'disabled': QCursor(QPixmap(path_cursor).scaled(20, 20)),
    #         'cross': Qt.CrossCursor,
    #         'forbidden': Qt.ForbiddenCursor,
    #         'pointing': Qt.PointingHandCursor,
    #         'standard': QCursor()
    #     }

class QtMarkersControls(QFrame):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        self.selectCheckBox = QtSelectCheckBox(layer)
        self.additionCheckBox = QtAdditionCheckBox(layer)

        layout = QVBoxLayout()
        layout.addWidget(self.selectCheckBox)
        layout.addWidget(self.additionCheckBox)
        layout.addStretch(0)
        self.setLayout(layout)
        #self.setFixedWidth(28)

class QtSelectCheckBox(QCheckBox):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
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
        self.setFixedWidth(28)

    def _set_mode(self, bool):
        if bool:
            self.layer.mode = 'select'
        else:
            self.layer.mode = None


class QtAdditionCheckBox(QCheckBox):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
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
        self.setFixedWidth(28)

    def _set_mode(self, bool):
        if bool:
            self.layer.mode = 'add'
        else:
            self.layer.mode = None
