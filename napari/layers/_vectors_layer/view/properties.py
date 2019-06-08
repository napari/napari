from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QLabel,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QGridLayout,
)
import numpy as np

from ..._base_layer import QtLayerProperties


class QtVectorsProperties(QtLayerProperties):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.averaging.connect(self._on_avg_change)
        self.layer.events.width.connect(self._on_width_change)
        self.layer.events.length.connect(self._on_len_change)

        # vector color adjustment and widget
        face_comboBox = QComboBox()
        colors = self.layer._colors
        for c in colors:
            face_comboBox.addItem(c)
        index = face_comboBox.findText(self.layer.color, Qt.MatchFixedString)
        if index >= 0:
            face_comboBox.setCurrentIndex(index)
        face_comboBox.activated[str].connect(
            lambda text=face_comboBox: self.change_face_color(text)
        )
        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(QLabel('color:'), row, self.name_column)
        self.grid_layout.addWidget(face_comboBox, row, self.property_column)

        # line width in pixels
        self.width_field = QDoubleSpinBox()
        self.width_field.setSingleStep(0.1)
        self.width_field.setMinimum(0.1)
        value = self.layer.width
        self.width_field.setValue(value)
        self.width_field.valueChanged.connect(self.change_width)
        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(QLabel('width:'), row, self.name_column)
        self.grid_layout.addWidget(self.width_field, row, self.property_column)

        # averaging spinbox
        self.averaging_spinbox = QSpinBox()
        self.averaging_spinbox.setSingleStep(1)
        self.averaging_spinbox.setValue(1)
        self.averaging_spinbox.setMinimum(1)
        self.averaging_spinbox.valueChanged.connect(self.change_average_type)
        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(QLabel('avg kernel'), row, self.name_column)
        self.grid_layout.addWidget(
            self.averaging_spinbox, row, self.property_column
        )

        # line length
        self.length_field = QDoubleSpinBox()
        self.length_field.setSingleStep(0.1)
        value = self.layer.length
        self.length_field.setValue(value)
        self.length_field.setMinimum(0.1)
        self.length_field.valueChanged.connect(self.change_length)
        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(QLabel('length:'), row, self.name_column)
        self.grid_layout.addWidget(
            self.length_field, row, self.property_column
        )

        self.setExpanded(False)

    def change_face_color(self, text):
        self.layer.color = text

    def change_connector_type(self, text):
        self.layer.connector = text

    def change_average_type(self, value):
        self.layer.averaging = value

    def change_width(self, value):
        self.layer.width = value

    def change_length(self, value):
        self.layer.length = value

    def _on_avg_change(self, event):
        with self.layer.events.averaging.blocker():
            self.averaging_spinbox.setValue(self.layer.averaging)

    def _on_len_change(self, event):
        with self.layer.events.length.blocker():
            self.length_field.setValue(self.layer.length)

    def _on_width_change(self, event):
        with self.layer.events.width.blocker():
            self.width_field.setValue(self.layer.width)
