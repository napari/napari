from collections.abc import Iterable
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QLabel,
    QComboBox,
    QSlider,
    QFrame,
    QGridLayout,
)
from vispy.color import Color
from .qt_base_layer import QtLayerControls
from ...layers.shapes._constants import Mode
from ..qt_mode_buttons import QtModeRadioButton, QtModePushButton


class QtShapesControls(QtLayerControls):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.mode.connect(self.set_mode)
        self.layer.events.edge_width.connect(self._on_edge_width_change)
        self.layer.events.edge_color.connect(self._on_edge_color_change)
        self.layer.events.face_color.connect(self._on_face_color_change)
        self.layer.events.editable.connect(self._on_editable_change)

        sld = QSlider(Qt.Horizontal)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setMinimum(0)
        sld.setMaximum(40)
        sld.setSingleStep(1)
        value = self.layer.current_edge_width
        if isinstance(value, Iterable):
            if isinstance(value, list):
                value = np.asarray(value)
            value = value.mean()
        sld.setValue(int(value))
        sld.valueChanged.connect(self.changeWidth)
        self.widthSlider = sld

        face_comboBox = QComboBox()
        face_comboBox.addItems(self.layer._colors)
        face_comboBox.activated[str].connect(self.changeFaceColor)
        self.faceComboBox = face_comboBox
        self.faceColorSwatch = QFrame()
        self.faceColorSwatch.setObjectName('swatch')
        self.faceColorSwatch.setToolTip('Face color swatch')
        self._on_face_color_change()

        edge_comboBox = QComboBox()
        edge_comboBox.addItems(self.layer._colors)
        edge_comboBox.activated[str].connect(self.changeEdgeColor)
        self.edgeComboBox = edge_comboBox
        self.edgeColorSwatch = QFrame()
        self.edgeColorSwatch.setObjectName('swatch')
        self.edgeColorSwatch.setToolTip('Edge color swatch')
        self._on_edge_color_change()

        self.select_button = QtModeRadioButton(
            layer, 'select', Mode.SELECT, tooltip='Select shapes'
        )
        self.direct_button = QtModeRadioButton(
            layer, 'direct', Mode.DIRECT, tooltip='Select vertices'
        )
        self.panzoom_button = QtModeRadioButton(
            layer, 'zoom', Mode.PAN_ZOOM, tooltip='Pan/zoom', checked=True
        )
        self.rectangle_button = QtModeRadioButton(
            layer, 'rectangle', Mode.ADD_RECTANGLE, tooltip='Add rectangles'
        )
        self.ellipse_button = QtModeRadioButton(
            layer, 'ellipse', Mode.ADD_ELLIPSE, tooltip='Add ellipses'
        )
        self.line_button = QtModeRadioButton(
            layer, 'line', Mode.ADD_LINE, tooltip='Add lines'
        )
        self.path_button = QtModeRadioButton(
            layer, 'path', Mode.ADD_PATH, tooltip='Add paths'
        )
        self.polygon_button = QtModeRadioButton(
            layer, 'polygon', Mode.ADD_POLYGON, tooltip='Add polygons'
        )
        self.vertex_insert_button = QtModeRadioButton(
            layer, 'vertex_insert', Mode.VERTEX_INSERT, tooltip='Insert vertex'
        )
        self.vertex_remove_button = QtModeRadioButton(
            layer, 'vertex_remove', Mode.VERTEX_REMOVE, tooltip='Remove vertex'
        )

        self.move_front_button = QtModePushButton(
            layer,
            'move_front',
            slot=self.layer.move_to_front,
            tooltip='Move to front',
        )
        self.move_back_button = QtModePushButton(
            layer,
            'move_back',
            slot=self.layer.move_to_back,
            tooltip='Move to back',
        )
        self.delete_button = QtModePushButton(
            layer,
            'delete_shape',
            slot=self.layer.remove_selected,
            tooltip='Delete selected shapes',
        )

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

        button_grid = QGridLayout()
        button_grid.addWidget(self.vertex_remove_button, 0, 1)
        button_grid.addWidget(self.vertex_insert_button, 0, 2)
        button_grid.addWidget(self.delete_button, 0, 3)
        button_grid.addWidget(self.direct_button, 0, 4)
        button_grid.addWidget(self.select_button, 0, 5)
        button_grid.addWidget(self.panzoom_button, 0, 6)
        button_grid.addWidget(self.move_back_button, 1, 0)
        button_grid.addWidget(self.move_front_button, 1, 1)
        button_grid.addWidget(self.ellipse_button, 1, 2)
        button_grid.addWidget(self.rectangle_button, 1, 3)
        button_grid.addWidget(self.polygon_button, 1, 4)
        button_grid.addWidget(self.line_button, 1, 5)
        button_grid.addWidget(self.path_button, 1, 6)
        button_grid.setColumnStretch(2, 2)
        button_grid.setSpacing(4)

        # grid_layout created in QtLayerControls
        # addWidget(widget, row, column, [row_span, column_span])
        self.grid_layout.addLayout(button_grid, 0, 0, 1, 3)
        self.grid_layout.addWidget(QLabel('opacity:'), 1, 0)
        self.grid_layout.addWidget(self.opacitySlider, 1, 1, 1, 2)
        self.grid_layout.addWidget(QLabel('edge width:'), 2, 0)
        self.grid_layout.addWidget(self.widthSlider, 2, 1, 1, 2)
        self.grid_layout.addWidget(QLabel('blending:'), 3, 0)
        self.grid_layout.addWidget(self.blendComboBox, 3, 1, 1, 2)
        self.grid_layout.addWidget(QLabel('face color:'), 4, 0)
        self.grid_layout.addWidget(self.faceComboBox, 4, 2)
        self.grid_layout.addWidget(self.faceColorSwatch, 4, 1)
        self.grid_layout.addWidget(QLabel('edge color:'), 5, 0)
        self.grid_layout.addWidget(self.edgeComboBox, 5, 2)
        self.grid_layout.addWidget(self.edgeColorSwatch, 5, 1)
        self.grid_layout.setRowStretch(6, 1)
        self.grid_layout.setColumnStretch(1, 1)
        self.grid_layout.setSpacing(4)

    def mouseMoveEvent(self, event):
        self.layer.status = str(self.layer.mode)

    def set_mode(self, event):
        mode_buttons = {
            Mode.SELECT: self.select_button,
            Mode.DIRECT: self.direct_button,
            Mode.PAN_ZOOM: self.panzoom_button,
            Mode.ADD_RECTANGLE: self.rectangle_button,
            Mode.ADD_ELLIPSE: self.ellipse_button,
            Mode.ADD_LINE: self.line_button,
            Mode.ADD_PATH: self.path_button,
            Mode.ADD_POLYGON: self.polygon_button,
            Mode.VERTEX_INSERT: self.vertex_insert_button,
            Mode.VERTEX_REMOVE: self.vertex_remove_button,
        }

        if event.mode in mode_buttons:
            mode_buttons[event.mode].setChecked(True)
        else:
            raise ValueError(f"Mode '{event.mode}'not recognized")

    def changeFaceColor(self, text):
        self.layer.current_face_color = text

    def changeEdgeColor(self, text):
        self.layer.current_edge_color = text

    def changeWidth(self, value):
        self.layer.current_edge_width = float(value) / 2

    def changeOpacity(self, value):
        with self.layer.events.blocker(self._on_opacity_change):
            self.layer.current_opacity = value / 100

    def _on_edge_width_change(self, event=None):
        with self.layer.events.edge_width.blocker():
            value = self.layer.current_edge_width
            value = np.clip(int(2 * value), 0, 40)
            self.widthSlider.setValue(value)

    def _on_edge_color_change(self, event=None):
        with self.layer.events.edge_color.blocker():
            index = self.edgeComboBox.findText(
                self.layer.current_edge_color, Qt.MatchFixedString
            )
            self.edgeComboBox.setCurrentIndex(index)
        color = Color(self.layer.current_edge_color).hex
        self.edgeColorSwatch.setStyleSheet("background-color: " + color)

    def _on_face_color_change(self, event=None):
        with self.layer.events.face_color.blocker():
            index = self.faceComboBox.findText(
                self.layer.current_face_color, Qt.MatchFixedString
            )
            self.faceComboBox.setCurrentIndex(index)
        color = Color(self.layer.current_face_color).hex
        self.faceColorSwatch.setStyleSheet("background-color: " + color)

    def _on_opacity_change(self, event=None):
        with self.layer.events.opacity.blocker():
            self.opacitySlider.setValue(self.layer.current_opacity * 100)

    def _on_editable_change(self, event=None):
        self.select_button.setEnabled(self.layer.editable)
        self.direct_button.setEnabled(self.layer.editable)
        self.rectangle_button.setEnabled(self.layer.editable)
        self.ellipse_button.setEnabled(self.layer.editable)
        self.line_button.setEnabled(self.layer.editable)
        self.path_button.setEnabled(self.layer.editable)
        self.polygon_button.setEnabled(self.layer.editable)
        self.vertex_remove_button.setEnabled(self.layer.editable)
        self.vertex_insert_button.setEnabled(self.layer.editable)
        self.delete_button.setEnabled(self.layer.editable)
        self.move_back_button.setEnabled(self.layer.editable)
        self.move_front_button.setEnabled(self.layer.editable)
