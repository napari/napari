from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QButtonGroup, QVBoxLayout, QRadioButton, QFrame

from os.path import join
from ....resources import resources_dir


class QtMarkersControls(QFrame):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        self.layer.events.mode.connect(self.set_mode)

        self.select_button = QtSelectButton(layer)
        self.addition_button = QtAdditionButton(layer)
        self.panzoom_button = QtPanZoomButton(layer)

        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.select_button)
        self.button_group.addButton(self.addition_button)
        self.button_group.addButton(self.panzoom_button)

        layout = QVBoxLayout()
        layout.addWidget(self.select_button)
        layout.addWidget(self.addition_button)
        layout.addWidget(self.panzoom_button)
        layout.addStretch(0)
        self.setLayout(layout)
        self.setMouseTracking(True)

    def mouseMoveEvent(self, event):
        self.layer.status = self.layer.mode

    def set_mode(self, event):
        mode = event.mode
        if mode == 'add':
            self.addition_button.setChecked(True)
        elif mode == 'select':
            self.select_button.setChecked(True)
        elif mode == 'pan/zoom':
            self.panzoom_button.setChecked(True)
        else:
            raise ValueError("Mode not recongnized")


class QtPanZoomButton(QRadioButton):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        self.setToolTip('Pan/zoom mode')
        self.setChecked(True)
        styleSheet = button_style('zoom')
        self.setStyleSheet(styleSheet)
        self.toggled.connect(lambda state=self: self._set_mode(state))
        self.setFixedWidth(28)

    def _set_mode(self, bool):
        with self.layer.events.mode.blocker():
            if bool:
                self.layer.mode = 'pan/zoom'


class QtSelectButton(QRadioButton):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        self.setToolTip('Select mode')
        self.setChecked(False)
        styleSheet = button_style('select')
        self.setStyleSheet(styleSheet)
        self.toggled.connect(lambda state=self: self._set_mode(state))
        self.setFixedWidth(28)

    def _set_mode(self, bool):
        with self.layer.events.mode.blocker():
            if bool:
                self.layer.mode = 'select'


class QtAdditionButton(QRadioButton):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        self.setToolTip('Addition mode')
        self.setChecked(False)
        styleSheet = button_style('add')
        self.setStyleSheet(styleSheet)
        self.toggled.connect(lambda state=self: self._set_mode(state))
        self.setFixedWidth(28)

    def _set_mode(self, bool):
        with self.layer.events.mode.blocker():
            if bool:
                self.layer.mode = 'add'


def button_style(name):
    path_off = join(resources_dir, 'icons', name + '_off.png')
    path_on = join(resources_dir, 'icons', name + '_on.png')
    return """QRadioButton {background-color:lightGray;
              border-radius: 3px;}
              QRadioButton::indicator {subcontrol-position:
              center center; subcontrol-origin: content;
              width: 28px; height: 28px;}
              QRadioButton::indicator:checked {
              background-color:rgb(0, 153, 255); border-radius: 3px;
              image: url(""" + path_off + """);}
              QRadioButton::indicator:unchecked
              {image: url(""" + path_off + """);}
              QRadioButton::indicator:unchecked:hover
              {image: url(""" + path_on + ");}"
