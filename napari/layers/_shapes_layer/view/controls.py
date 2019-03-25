from enum import Enum

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QButtonGroup, QVBoxLayout, QRadioButton, QFrame,
                             QPushButton)

from os.path import join
from ....resources import resources_dir


class Mode(Enum):
    PanZoom = 0
    Select = 1
    Direct = 2
    AddRectangle = 3
    AddEllipse = 4
    AddLine = 5
    AddPath = 6
    AddPolygon = 7
    VertexInsert = 8
    VertexRemove = 9


class QtShapesControls(QFrame):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        self.layer.events.mode.connect(self.set_mode)

        self.select_button = QtModeButton(layer, 'select', Mode.Select,
                                          'Select mode')
        self.direct_button = QtModeButton(layer, 'direct', Mode.Direct,
                                          'Direct select mode')
        self.panzoom_button = QtModeButton(layer, 'zoom', Mode.PanZoom,
                                           'Pan/zoom mode')
        self.rectangle_button = QtModeButton(layer, 'rectangle',
                                             Mode.AddRectangle,
                                             'Add rectangles')
        self.ellipse_button = QtModeButton(layer, 'ellipse', Mode.AddEllipse,
                                           'Add ellipses')
        self.line_button = QtModeButton(layer, 'line', Mode.AddLine,
                                        'Add lines')
        self.path_button = QtModeButton(layer, 'path', Mode.AddPath,
                                        'Add paths')
        self.polygon_button = QtModeButton(layer, 'polygon', Mode.AddPolygon,
                                           'Add polygons')
        self.vertex_insert_button = QtModeButton(layer, 'vertex_insert',
                                                 Mode.VertexInsert,
                                                 'Insert vertex')
        self.vertex_remove_button = QtModeButton(layer, 'vertex_remove',
                                                 Mode.VertexRemove,
                                                 'Remove vertex')

        self.move_front_button = QtMoveFrontButton(layer)
        self.move_back_button = QtMoveBackButton(layer)
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
        layout.addWidget(self.move_front_button)
        layout.addWidget(self.move_back_button)
        layout.addWidget(self.delete_button)
        layout.addStretch(0)
        self.setLayout(layout)
        self.setMouseTracking(True)

        self.panzoom_button.setChecked(True)

    def mouseMoveEvent(self, event):
        self.layer.status = str(self.layer.mode)

    def set_mode(self, event):
        mode = event.mode
        if mode == Mode.Select:
            self.select_button.setChecked(True)
        elif mode == Mode.Direct:
            self.direct_button.setChecked(True)
        elif mode == Mode.PanZoom:
            self.panzoom_button.setChecked(True)
        elif mode == Mode.AddRectangle:
            self.rectangle_button.setChecked(True)
        elif mode == Mode.AddEllipse:
            self.ellipse_button.setChecked(True)
        elif mode == Mode.AddLine:
            self.line_button.setChecked(True)
        elif mode == Mode.AddPath:
            self.path_button.setChecked(True)
        elif mode == Mode.AddPolygon:
            self.polygon_button.setChecked(True)
        elif mode == Mode.VertexInsert:
            self.vertex_insert_button.setChecked(True)
        elif mode == Mode.VertexRemove:
            self.vertex_remove_button.setChecked(True)
        else:
            raise ValueError("Mode not recongnized")


class QtModeButton(QRadioButton):
    def __init__(self, layer, button_name, mode, tool_tip):
        super().__init__()

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

        self.layer = layer
        self.setIcon(QIcon(':/icons/delete.png'))
        self.setFixedWidth(28)
        self.setFixedHeight(28)
        self.setToolTip('Delete selected')
        self.clicked.connect(self.layer.remove_selected)


class QtMoveBackButton(QPushButton):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        self.setIcon(QIcon(':/icons/move_back.png'))
        self.setFixedWidth(28)
        self.setFixedHeight(28)
        self.setToolTip('Move to back')
        self.clicked.connect(self.layer.move_to_back)


class QtMoveFrontButton(QPushButton):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        self.setIcon(QIcon(':/icons/move_front.png'))
        self.setFixedWidth(28)
        self.setFixedHeight(28)
        self.setToolTip('Move to front')
        self.clicked.connect(self.layer.move_to_front)


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
