from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QButtonGroup, QVBoxLayout, QRadioButton, QFrame


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
        self.toggled.connect(lambda state=self: self._set_mode(state))
        self.setFixedWidth(28)

    def _set_mode(self, bool):
        with self.layer.events.mode.blocker():
            if bool:
                self.layer.mode = 'add'
