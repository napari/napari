from qtpy.QtWidgets import QRadioButton, QPushButton


class QtModeRadioButton(QRadioButton):
    def __init__(self, layer, button_name, mode, tool_tip, checked=False):
        super().__init__()

        self.mode = mode
        self.layer = layer
        self.setToolTip(tool_tip)
        self.setChecked(checked)
        self.setProperty('mode', button_name)
        self.toggled.connect(self._set_mode)
        self.setFixedWidth(28)

    def _set_mode(self, bool):
        with self.layer.events.mode.blocker(self._set_mode):
            if bool:
                self.layer.mode = self.mode


class QtModePushButton(QPushButton):
    def __init__(self, layer, button_name, tool_tip, slot=None):
        super().__init__()

        self.layer = layer
        self.setProperty('mode', button_name)
        self.setToolTip(tool_tip)
        self.setFixedWidth(28)
        self.setFixedHeight(28)
        if slot is not None:
            self.clicked.connect(slot)
