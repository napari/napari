from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QButtonGroup, QVBoxLayout, QRadioButton, QFrame, QPushButton

from os.path import join
from ....resources import resources_dir


class QtShapesControls(QFrame):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        self.layer.events.mode.connect(self.set_mode)

        self.select_button = QtModeButton(layer, 'select', 'Select mode')
        self.direct_button = QtModeButton(layer, 'direct', 'Direct select mode')
        self.panzoom_button = QtModeButton(layer, 'zoom', 'Pan/zoom mode', mode='pan/zoom')
        self.rectangle_button = QtModeButton(layer, 'rectangle', 'Add rectangles', mode='add_rectangle')
        self.ellipse_button = QtModeButton(layer, 'ellipse', 'Add ellipses', mode='add_ellipse')
        self.line_button = QtModeButton(layer, 'line', 'Add lines', mode='add_line')
        self.path_button = QtModeButton(layer, 'path', 'Add paths', mode='add_path')
        self.polygon_button = QtModeButton(layer, 'polygon', 'Add polygons', mode='add_polygon')
        self.vertex_insert_button = QtModeButton(layer, 'vertex_insert', 'Insert vertex')
        self.vertex_remove_button = QtModeButton(layer, 'vertex_remove', 'Remove vertex')
        self.delete_button = QtDeleteButton(layer)

        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.select_button)
        self.button_group.addButton(self.direct_button)
        self.button_group.addButton(self.panzoom_button)
        self.button_group.addButton(self.rectangle_button)
        self.button_group.addButton(self.ellipse_button)
        self.button_group.addButton(self.line_button)
        self.button_group.addButton(self.path_button)
        self.button_group.addButton(self.polygon_button)
        self.button_group.addButton(self.vertex_insert_button)
        self.button_group.addButton(self.vertex_remove_button)

        layout = QVBoxLayout()
        layout.addWidget(self.panzoom_button)
        layout.addWidget(self.select_button)
        layout.addWidget(self.direct_button)
        layout.addWidget(self.vertex_insert_button)
        layout.addWidget(self.vertex_remove_button)
        layout.addWidget(self.rectangle_button)
        layout.addWidget(self.ellipse_button)
        layout.addWidget(self.line_button)
        layout.addWidget(self.path_button)
        layout.addWidget(self.polygon_button)
        layout.addWidget(self.delete_button)
        layout.addStretch(0)
        self.setLayout(layout)
        self.setMouseTracking(True)

        self.panzoom_button.setChecked(True)

    def mouseMoveEvent(self, event):
        self.layer.status = self.layer.mode

    def set_mode(self, event):
        mode = event.mode
        if mode == 'select':
            self.select_button.setChecked(True)
        elif mode == 'direct':
            self.direct_button.setChecked(True)
        elif mode == 'pan/zoom':
            self.panzoom_button.setChecked(True)
        elif mode == 'add_rectangle':
            self.rectangle_button.setChecked(True)
        elif mode == 'add_ellipse':
            self.ellipse_button.setChecked(True)
        elif mode == 'add_line':
            self.line_button.setChecked(True)
        elif mode == 'add_path':
            self.path_button.setChecked(True)
        elif mode == 'add_polygon':
            self.polygon_button.setChecked(True)
        elif mode == 'vertex_insert':
            self.vertex_insert_button.setChecked(True)
        elif mode == 'vertex_remove':
            self.vertex_remove_button.setChecked(True)
        else:
            raise ValueError("Mode not recongnized")


class QtModeButton(QRadioButton):
    def __init__(self, layer, button_name, tool_tip, mode=None):
        super().__init__()

        if mode is None:
            self.mode = button_name
        else:
            self.mode = mode
        self.layer = layer
        self.setToolTip(tool_tip)
        self.setChecked(False)
        styleSheet = button_style(button_name)
        self.setStyleSheet(styleSheet)
        self.toggled.connect(lambda state=self: self._set_mode(state))
        self.setFixedWidth(28)

    def _set_mode(self, bool):
        with self.layer.events.mode.blocker(self._set_mode):
            if bool:
                self.layer.mode = self.mode


class QtDeleteButton(QPushButton):
    def __init__(self, layer):
        super().__init__()

        path_delete = join(resources_dir, 'icons', 'delete.png')

        self.layer = layer
        self.setIcon(QIcon(path_delete))
        self.setFixedWidth(28)
        self.setFixedHeight(28)
        self.setToolTip('Delete selected')
        self.setChecked(False)
        self.setStyleSheet(styleSheet)
        self.clicked.connect(self.layer.remove_selected)


styleSheet = """QPushButton {background-color:lightGray; border-radius: 3px;}
                QPushButton:pressed {background-color:rgb(0, 153, 255);
                border-radius: 3px;}
                QPushButton:hover {background-color:rgb(0, 153, 255);
                border-radius: 3px;}"""


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
