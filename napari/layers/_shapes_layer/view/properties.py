from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QComboBox, QSlider, QCheckBox
from collections import Iterable
import numpy as np
from ..._base_layer import QtLayer


class QtShapesLayer(QtLayer):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.edge_width.connect(self._on_edge_width_change)
        self.layer.events.edge_color.connect(self._on_edge_color_change)
        self.layer.events.face_color.connect(self._on_face_color_change)
        self.layer.events.broadcast.connect(self._on_broadcast_change)

        sld = QSlider(Qt.Horizontal, self)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setFixedWidth(75)
        sld.setMinimum(0)
        sld.setMaximum(40)
        sld.setSingleStep(1)
        value = self.layer.edge_width
        if isinstance(value, Iterable):
            if isinstance(value, list):
                value = np.asarray(value)
            value = value.mean()
        sld.setValue(int(value))
        sld.valueChanged[int].connect(lambda value=sld:
                                      self.changeWidth(value))
        self.widthSlider = sld
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
        face_comboBox.activated[str].connect(lambda text=face_comboBox:
                                             self.changeFaceColor(text))
        self.faceComboBox = face_comboBox
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
        edge_comboBox.activated[str].connect(lambda text=edge_comboBox:
                                             self.changeEdgeColor(text))
        self.edgeComboBox = edge_comboBox
        self.grid_layout.addWidget(QLabel('edge_color:'), 5, 0)
        self.grid_layout.addWidget(edge_comboBox, 5, 1)

        broadcast_cb = QCheckBox()
        broadcast_cb.setToolTip('broadcast shapes')
        broadcast_cb.setChecked(self.layer.broadcast)
        broadcast_cb.stateChanged.connect(lambda state=broadcast_cb:
                                          self.change_broadcast(state))
        self.broadcastCheckBox = broadcast_cb
        self.grid_layout.addWidget(QLabel('broadcast:'), 6, 0)
        self.grid_layout.addWidget(broadcast_cb, 6, 1)

        self.setExpanded(False)

    def changeFaceColor(self, text):
        self.layer.face_color = text

    def changeEdgeColor(self, text):
        self.layer.edge_color = text

    def changeWidth(self, value):
        self.layer.edge_width = float(value)/2

    def change_broadcast(self, state):
        if state == Qt.Checked:
            self.layer.broadcast = True
        else:
            self.layer.broadcast = False

    def _on_edge_width_change(self, event):
        with self.layer.events.edge_width.blocker():
            value = self.layer.edge_width
            value = np.clip(int(2*value), 0, 40)
            self.widthSlider.setValue(value)

    def _on_edge_color_change(self, event):
        with self.layer.events.edge_color.blocker():
            index = self.edgeComboBox.findText(self.layer.edge_color,
                                               Qt.MatchFixedString)
            self.edgeComboBox.setCurrentIndex(index)

    def _on_face_color_change(self, event):
        with self.layer.events.face_color.blocker():
            index = self.faceComboBox.findText(self.layer.face_color,
                                               Qt.MatchFixedString)
            self.faceComboBox.setCurrentIndex(index)

    def _on_broadcast_change(self, event):
        with self.layer.events.broadcast.blocker():
            self.broadcastCheckBox.setChecked(self.layer.broadcast)
