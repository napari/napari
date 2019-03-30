from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QButtonGroup, QVBoxLayout, QRadioButton, QFrame,
                             QPushButton)

from .._constants import Mode


class QtLabelsControls(QFrame):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        self.layer.events.mode.connect(self.set_mode)

        self.panzoom_button = QtModeButton(layer, 'zoom', Mode.PAN_ZOOM,
                                           'Pan/zoom mode')
        self.pick_button = QtModeButton(layer, 'pick', Mode.PICK,
                                          'Pick mode')
        self.paint_button = QtModeButton(layer, 'paint', Mode.PAINT,
                                          'Paint mode')
        self.fill_button = QtModeButton(layer, 'fill', Mode.FILL,
                                          'Fill mode')

        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.panzoom_button)
        self.button_group.addButton(self.pick_button)
        self.button_group.addButton(self.paint_button)
        self.button_group.addButton(self.fill_button)

        layout = QVBoxLayout()
        layout.addWidget(self.panzoom_button)
        layout.addWidget(self.pick_button)
        layout.addWidget(self.paint_button)
        layout.addWidget(self.fill_button)
        layout.addStretch(0)
        self.setLayout(layout)
        self.setMouseTracking(True)

        self.panzoom_button.setChecked(True)

    def mouseMoveEvent(self, event):
        self.layer.status = str(self.layer.mode)

    def set_mode(self, event):
        mode = event.mode
        if mode == Mode.PAN_ZOOM:
            self.panzoom_button.setChecked(True)
        elif mode == Mode.PICK:
            self.pick_button.setChecked(True)
        elif mode == Mode.PAINT:
            self.paint_button.setChecked(True)
        elif mode == Mode.FILL:
            self.fill_button.setChecked(True)
        else:
            raise ValueError("Mode not recongnized")


class QtModeButton(QRadioButton):
    def __init__(self, layer, button_name, mode, tool_tip):
        super().__init__()

        self.mode = mode
        self.layer = layer
        self.setToolTip(tool_tip)
        self.setChecked(False)
        self.setProperty('mode', button_name)
        self.toggled.connect(lambda state=self: self._set_mode(state))
        self.setFixedWidth(28)

    def _set_mode(self, bool):
        with self.layer.events.mode.blocker(self._set_mode):
            if bool:
                self.layer.mode = self.mode
