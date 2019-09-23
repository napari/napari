from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QComboBox, QDoubleSpinBox
from .qt_base_layer import QtLayerControls


class QtVectorsControls(QtLayerControls):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.edge_width.connect(self._on_width_change)
        self.layer.events.length.connect(self._on_len_change)

        # vector color adjustment and widget
        face_comboBox = QComboBox()
        colors = self.layer._colors
        for c in colors:
            face_comboBox.addItem(c)
        index = face_comboBox.findText(
            self.layer.edge_color, Qt.MatchFixedString
        )
        if index >= 0:
            face_comboBox.setCurrentIndex(index)
        face_comboBox.activated[str].connect(
            lambda text=face_comboBox: self.change_edge_color(text)
        )
        self.faceComboBox = face_comboBox

        # line width in pixels
        self.widthSpinBox = QDoubleSpinBox()
        self.widthSpinBox.setKeyboardTracking(False)
        self.widthSpinBox.setSingleStep(0.1)
        self.widthSpinBox.setMinimum(0.1)
        value = self.layer.edge_width
        self.widthSpinBox.setValue(value)
        self.widthSpinBox.valueChanged.connect(self.change_width)

        # line length
        self.lengthSpinBox = QDoubleSpinBox()
        self.lengthSpinBox.setKeyboardTracking(False)
        self.lengthSpinBox.setSingleStep(0.1)
        value = self.layer.length
        self.lengthSpinBox.setValue(value)
        self.lengthSpinBox.setMinimum(0.1)
        self.lengthSpinBox.valueChanged.connect(self.change_length)

        layout_option = 2
        if layout_option == 1:
            self.grid_layout.addWidget(QLabel('opacity:'), 0, 0, 1, 4)
            self.grid_layout.addWidget(self.opacitySilder, 1, 0, 1, 4)
            self.grid_layout.addWidget(QLabel('width:'), 2, 0, 1, 4)
            self.grid_layout.addWidget(self.widthSpinBox, 3, 0, 1, 4)
            self.grid_layout.addWidget(QLabel('length:'), 4, 0, 1, 4)
            self.grid_layout.addWidget(self.lengthSpinBox, 5, 0, 1, 4)
            self.grid_layout.addWidget(QLabel('face color:'), 6, 0, 1, 4)
            self.grid_layout.addWidget(self.faceComboBox, 7, 0, 1, 4)
            self.grid_layout.addWidget(QLabel('blending:'), 8, 0, 1, 3)
            self.grid_layout.addWidget(self.blendComboBox, 9, 0, 1, 3)
            self.grid_layout.setRowStretch(10, 1)
        elif layout_option == 2:
            self.grid_layout.addWidget(QLabel('opacity:'), 0, 0, 1, 3)
            self.grid_layout.addWidget(self.opacitySilder, 0, 3, 1, 4)
            self.grid_layout.addWidget(QLabel('width:'), 1, 0, 1, 0)
            self.grid_layout.addWidget(self.widthSpinBox, 1, 3, 1, 4)
            self.grid_layout.addWidget(QLabel('length:'), 2, 0, 1, 3)
            self.grid_layout.addWidget(self.lengthSpinBox, 2, 3, 1, 4)
            self.grid_layout.addWidget(QLabel('face color:'), 3, 0, 1, 3)
            self.grid_layout.addWidget(self.faceComboBox, 3, 3, 1, 4)
            self.grid_layout.addWidget(QLabel('blending:'), 4, 0, 1, 3)
            self.grid_layout.addWidget(self.blendComboBox, 4, 3, 1, 4)
            self.grid_layout.setRowStretch(5, 1)
            self.grid_layout.setVerticalSpacing(4)

    def change_edge_color(self, text):
        self.layer.edge_color = text

    def change_connector_type(self, text):
        self.layer.connector = text

    def change_width(self, value):
        self.layer.edge_width = value
        self.widthSpinBox.clearFocus()
        self.setFocus()

    def change_length(self, value):
        self.layer.length = value
        self.lengthSpinBox.clearFocus()
        self.setFocus()

    def _on_len_change(self, event):
        with self.layer.events.length.blocker():
            self.lengthSpinBox.setValue(self.layer.length)

    def _on_width_change(self, event):
        with self.layer.events.edge_width.blocker():
            self.widthSpinBox.setValue(self.layer.edge_width)
