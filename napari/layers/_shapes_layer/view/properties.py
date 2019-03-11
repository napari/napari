from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QComboBox, QSlider, QCheckBox
from collections import Iterable
import numpy as np
from ..._base_layer import QtLayer


class QtShapesLayer(QtLayer):
    def __init__(self, layer):
        super().__init__(layer)

        sld = QSlider(Qt.Horizontal, self)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setFixedWidth(75)
        sld.setMinimum(0)
        sld.setMaximum(100)
        sld.setSingleStep(1)
        value = self.layer.edge_width
        if isinstance(value, Iterable):
            if isinstance(value, list):
                value = np.asarray(value)
            value = value.mean()
        sld.setValue(int(value))
        sld.valueChanged[int].connect(lambda value=sld: self.changeWidth(value))
        self.grid_layout.addWidget(QLabel('width:'), 3, 0)
        self.grid_layout.addWidget(sld, 3, 1)

        face_comboBox = QComboBox()
        colors = self.layer._colors
        for c in colors:
            face_comboBox.addItem(c)
        index = face_comboBox.findText(
            self.layer.face_color, Qt.MatchFixedString)
        if index >= 0:
            face_comboBox.setCurrentIndex(index)
        face_comboBox.activated[str].connect(
            lambda text=face_comboBox: self.changeFaceColor(text))
        self.grid_layout.addWidget(QLabel('face_color:'), 4, 0)
        self.grid_layout.addWidget(face_comboBox, 4, 1)

        edge_comboBox = QComboBox()
        colors = self.layer._colors
        for c in colors:
            edge_comboBox.addItem(c)
        index = edge_comboBox.findText(
            self.layer.edge_color, Qt.MatchFixedString)
        if index >= 0:
            edge_comboBox.setCurrentIndex(index)
        edge_comboBox.activated[str].connect(
            lambda text=edge_comboBox: self.changeEdgeColor(text))
        self.grid_layout.addWidget(QLabel('edge_color:'), 5, 0)
        self.grid_layout.addWidget(edge_comboBox, 5, 1)

        rearrange_comboBox = QComboBox()
        options = (['select', 'move_to_front', 'move_to_back',
                    'move_forward', 'move_backward'])
        for o in options:
            rearrange_comboBox.addItem(o)
        rearrange_comboBox.setCurrentIndex(0)
        rearrange_comboBox.activated[str].connect(
            lambda text=rearrange_comboBox: self.changeRearrange(text))
        self.rearrange_cb = rearrange_comboBox
        self.grid_layout.addWidget(QLabel('rearrange:'), 6, 0)
        self.grid_layout.addWidget(rearrange_comboBox, 6, 1)

        apply_cb = QCheckBox()
        apply_cb.setToolTip('Apply to all')
        apply_cb.setChecked(self.layer.apply_all)
        apply_cb.stateChanged.connect(lambda state=apply_cb:
                                     self.change_apply(state))
        self.grid_layout.addWidget(QLabel('apply_all:'), 7, 0)
        self.grid_layout.addWidget(apply_cb, 7, 1)

        self.setExpanded(False)

    def changeFaceColor(self, text):
        self.layer.face_color = text

    def changeEdgeColor(self, text):
        self.layer.edge_color = text

    def changeWidth(self, value):
        self.layer.edge_width = value

    def changeRearrange(self, text):
        if text == 'select':
            return
            
        _selected_shapes = self.layer._selected_shapes
        if len(_selected_shapes) == 0:
            self.rearrange_cb.setCurrentIndex(0)
            return

        if text == 'move_to_front':
            self.layer.move_to_front(_selected_shapes)
            self.rearrange_cb.setCurrentIndex(0)
        elif text == 'move_to_back':
            self.layer.move_to_back(_selected_shapes)
            self.rearrange_cb.setCurrentIndex(0)
        elif text == 'move_forward':
            self.layer.move_forward(_selected_shapes)
            self.rearrange_cb.setCurrentIndex(0)
        elif text == 'move_backward':
            self.layer.move_backward(_selected_shapes)
            self.rearrange_cb.setCurrentIndex(0)

    def change_apply(self, state):
        if state == Qt.Checked:
            self.layer.apply_all = True
        else:
            self.layer.apply_all = False
