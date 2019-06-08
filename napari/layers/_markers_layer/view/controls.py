from qtpy.QtWidgets import QButtonGroup, QVBoxLayout, QRadioButton, QFrame
from ..._base_layer import QtLayerControls
from .._constants import Mode


class QtMarkersControls(QtLayerControls):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.mode.connect(self.set_mode)

        self.select_button = QtSelectButton(layer)
        self.addition_button = QtAdditionButton(layer)
        self.panzoom_button = QtPanZoomButton(layer)

        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.select_button)
        self.button_group.addButton(self.addition_button)
        self.button_group.addButton(self.panzoom_button)

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 20, 10, 10)
        layout.addWidget(self.panzoom_button)
        layout.addWidget(self.select_button)
        layout.addWidget(self.addition_button)
        layout.addStretch(0)
        self.setLayout(layout)
        self.setMouseTracking(True)

    def mouseMoveEvent(self, event):
        self.layer.status = self.layer.mode

    def set_mode(self, event):
        mode = event.mode
        if mode == Mode.ADD:
            self.addition_button.setChecked(True)
        elif mode == Mode.SELECT:
            self.select_button.setChecked(True)
        elif mode == Mode.PAN_ZOOM:
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
                self.layer.mode = Mode.PAN_ZOOM


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
                self.layer.mode = Mode.SELECT


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
                self.layer.mode = Mode.ADD
