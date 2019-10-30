from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QLabel,
    QComboBox,
    QSlider,
    QCheckBox,
    QButtonGroup,
    QVBoxLayout,
    QRadioButton,
    QPushButton,
    QLineEdit,
    QFrame,
)
from vispy.color import Color
from .qt_base_layer import QtLayerControls
from ...layers.text._constants import Mode


class QtTextControls(QtLayerControls):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.mode.connect(self.set_mode)
        self.layer.events.rotation.connect(self._on_rotation_change)
        self.layer.events.font_size.connect(self._on_size_change)
        self.layer.events.text_color.connect(self._on_text_color_change)
        self.layer.events.editable.connect(self._on_editable_change)
        self.layer.events.highlight.connect(self._on_highlight)

        sld = QSlider(Qt.Horizontal)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setMinimum(1)
        sld.setMaximum(100)
        sld.setSingleStep(1)
        value = self.layer.font_size
        sld.setValue(int(value))
        sld.valueChanged[int].connect(lambda value=sld: self.changeSize(value))
        self.sizeSlider = sld

        rot_sld = QSlider(Qt.Horizontal)
        rot_sld.setFocusPolicy(Qt.NoFocus)
        rot_sld.setMinimum(-180)
        rot_sld.setMaximum(180)
        rot_sld.setSingleStep(1)
        angle = self.layer.rotation
        rot_sld.setValue(int(angle))
        rot_sld.valueChanged[int].connect(
            lambda value=sld: self.changeRotation(value)
        )
        self.rotationSlider = rot_sld

        color_comboBox = QComboBox()
        colors = self.layer._colors
        for c in colors:
            color_comboBox.addItem(c)
        color_comboBox.activated[str].connect(
            lambda text=color_comboBox: self.changeTextColor(text)
        )
        self.colorComboBox = color_comboBox
        self.textColorSwatch = QFrame()
        self.textColorSwatch.setObjectName('swatch')
        self.textColorSwatch.setToolTip('Text color swatch')
        self._on_text_color_change(None)

        text_box = QLineEdit()
        text_box.returnPressed.connect(
            lambda text=text_box: self.modify_text(text)
        )
        text_box.textChanged.connect(
            lambda text=text_box: self.change_text(text)
        )
        self.text_box = text_box

        self.select_button = QtSelectButton(layer)
        self.addition_button = QtAdditionButton(layer)
        self.panzoom_button = QtPanZoomButton(layer)
        self.delete_button = QtDeletePointsButton(layer)

        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.select_button)
        self.button_group.addButton(self.addition_button)
        self.button_group.addButton(self.panzoom_button)

        # grid_layout created in QtLayerControls
        # addWidget(widget, row, column, [row_span, column_span])
        self.grid_layout.addWidget(self.panzoom_button, 0, 6)
        self.grid_layout.addWidget(self.select_button, 0, 5)
        self.grid_layout.addWidget(self.addition_button, 0, 4)
        self.grid_layout.addWidget(self.delete_button, 0, 3)
        self.grid_layout.addWidget(QLabel('opacity:'), 1, 0, 1, 3)
        self.grid_layout.addWidget(self.opacitySilder, 1, 3, 1, 4)
        self.grid_layout.addWidget(QLabel('text size:'), 2, 0, 1, 3)
        self.grid_layout.addWidget(self.sizeSlider, 2, 3, 1, 4)
        self.grid_layout.addWidget(QLabel('text angle:'), 3, 0, 1, 3)
        self.grid_layout.addWidget(self.rotationSlider, 3, 3, 1, 4)
        self.grid_layout.addWidget(QLabel('blending:'), 4, 0, 1, 3)
        self.grid_layout.addWidget(self.blendComboBox, 4, 3, 1, 4)
        self.grid_layout.addWidget(QLabel('text color:'), 5, 0, 1, 3)
        self.grid_layout.addWidget(self.colorComboBox, 5, 3, 1, 3)
        self.grid_layout.addWidget(self.textColorSwatch, 5, 6)
        self.grid_layout.addWidget(QLabel('text:'), 6, 0, 1, 3)
        self.grid_layout.addWidget(self.text_box, 6, 3, 1, 6)
        self.grid_layout.setRowStretch(7, 1)
        self.grid_layout.setVerticalSpacing(4)

    def mouseMoveEvent(self, event):
        self.layer.status = self.layer.mode

    def set_mode(self, event):
        mode = event.mode
        if mode == Mode.ADD:
            self.addition_button.setChecked(True)
            self.text_box.setText('')
        elif mode == Mode.SELECT:
            self.select_button.setChecked(True)
            self._on_highlight(event=None)
        elif mode == Mode.PAN_ZOOM:
            self.panzoom_button.setChecked(True)
        else:
            raise ValueError("Mode not recongnized")

    def changeTextColor(self, text):
        self.layer.text_color = text

    def changeSize(self, value):
        self.layer.font_size = value

    def changeRotation(self, value):
        self.layer.rotation = value

    def modify_text(self, text):
        if self.layer._mode == Mode.ADD or self.layer._mode == Mode.SELECT:
            self.text_box.clearFocus()
            self.sizeSlider.setFocus()

    def change_text(self, text):
        if self.layer._mode == Mode.ADD:
            self.layer.new_text = text
        elif self.layer._mode == Mode.SELECT:
            selected_data = self.layer.selected_data
            data = self.layer.data
            if len(selected_data) > 0:
                for i in selected_data:
                    data[1][i] = text
                self.layer.data = data

    def _on_size_change(self, event):
        with self.layer.events.font_size.blocker():
            value = self.layer.font_size
            self.sizeSlider.setValue(int(value))

    def _on_rotation_change(self, event):
        with self.layer.events.rotation.blocker():
            value = self.layer.rotation
            self.rotationSlider.setValue(int(value))

    def _on_text_color_change(self, event):
        with self.layer.events.text_color.blocker():
            index = self.colorComboBox.findText(
                self.layer.text_color, Qt.MatchFixedString
            )
            self.colorComboBox.setCurrentIndex(index)
        color = Color(self.layer.text_color).hex
        self.textColorSwatch.setStyleSheet("background-color: " + color)

    def _on_editable_change(self, event):
        self.select_button.setEnabled(self.layer.editable)
        self.addition_button.setEnabled(self.layer.editable)
        self.delete_button.setEnabled(self.layer.editable)

    def _on_highlight(self, event):
        selected_data = self.layer.selected_data
        if len(selected_data) == 1:
            text = self.layer.text[selected_data[0]]
            self.text_box.setText(text)
        elif self.layer._mode == Mode.ADD:
            return


class QtPanZoomButton(QRadioButton):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        self.setToolTip('Pan/zoom')
        self.setChecked(True)
        self.toggled.connect(lambda state=self: self._set_mode(state))
        self.setFixedWidth(28)

    def _set_mode(self, bool):
        with self.layer.events.mode.blocker():
            if bool:
                self.layer.mode = Mode.PAN_ZOOM


class QtSelectButton(QRadioButton):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        self.setToolTip('Select points')
        self.setChecked(False)
        self.toggled.connect(lambda state=self: self._set_mode(state))
        self.setFixedWidth(28)

    def _set_mode(self, bool):
        with self.layer.events.mode.blocker():
            if bool:
                self.layer.mode = Mode.SELECT


class QtAdditionButton(QRadioButton):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        self.setToolTip('Add points')
        self.setChecked(False)
        self.toggled.connect(lambda state=self: self._set_mode(state))
        self.setFixedWidth(28)

    def _set_mode(self, bool):
        with self.layer.events.mode.blocker():
            if bool:
                self.layer.mode = Mode.ADD


class QtDeletePointsButton(QPushButton):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        self.setFixedWidth(28)
        self.setFixedHeight(28)
        self.setToolTip('Delete selected points')
        self.clicked.connect(self.layer.remove_selected)
