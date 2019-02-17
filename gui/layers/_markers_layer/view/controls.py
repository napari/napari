from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QButtonGroup, QVBoxLayout, QRadioButton, QFrame

from os.path import join
from ....resources import resources_dir


class QtMarkersControls(QFrame):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
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
        self.layer.viewer.status = self.layer.status

    # def on_key_press(self, event):
    #     if event.native.isAutoRepeat():
    #         return
    #     else:
    #         if event.key == ' ':
    #             if self.viewer.mode is not None:
    #                 self.viewer._mode_history = self.viewer.mode
    #                 self.viewer.mode = None
    #             else:
    #                 self.viewer._mode_history = None
    #         elif event.key == 'Meta':
    #             if self.viewer.mode == 'select' and self.viewer.active_markers:
    #                 self.canvas.native.setCursor(self._cursors['remove'])
    #                 layer = self.viewer.layers[self.viewer.active_markers]
    #                 layer.interact(self.viewer.position, self.viewer.dimensions.indices,
    #                 mode=self.viewer.mode, dragging=False, shift=False, ctrl=True,
    #                 pressed=False, released=False, moving=False)
    #         elif event.key == 'Shift':
    #             if self.viewer.mode is not None and self.viewer.active_markers:
    #                 layer = self.viewer.layers[self.viewer.active_markers]
    #                 layer.interact(self.viewer.position, self.viewer.dimensions.indices,
    #                 mode=self.viewer.mode, dragging=False, shift=True, ctrl=False,
    #                 pressed=False, released=False, moving=False)
    #         elif event.key == 'a':
    #             self.viewer._set_mode('add')
    #         elif event.key == 's':
    #             self.viewer._set_mode('select')
    #         elif event.key == 'n':
    #             self.viewer._set_mode(None)
    #
    # def on_key_release(self, event):
    #     if event.key == ' ':
    #         if self.viewer._mode_history is not None:
    #             self.viewer.mode = self.viewer._mode_history
    #     elif event.key == 'Meta':
    #         if self.viewer.mode == 'select' and self.viewer.active_markers:
    #             self.canvas.native.setCursor(self._cursors['pointing'])
    #             layer = self.viewer.layers[self.viewer.active_markers]
    #             layer.interact(self.viewer.position, self.viewer.dimensions.indices,
    #             mode=self.viewer.mode, dragging=False,
    #             shift=False, ctrl=False, pressed=False, released=False, moving=False)
    #     elif event.key == 'Shift':
    #         if self.viewer.mode is not None and self.viewer.active_markers:
    #             layer = self.viewer.layers[self.viewer.active_markers]
    #             layer.interact(self.viewer.position, self.viewer.dimensions.indices,
    #             mode=self.viewer.mode, dragging=False, shift=False, ctrl=False,
    #             pressed=False, released=False, moving=False)


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
