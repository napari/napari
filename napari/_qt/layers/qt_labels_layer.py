from qtpy.QtGui import QPainter, QColor
from qtpy.QtWidgets import (
    QButtonGroup,
    QVBoxLayout,
    QRadioButton,
    QWidget,
    QPushButton,
    QSlider,
    QCheckBox,
    QLabel,
    QSpinBox,
)
from qtpy.QtCore import Qt

import numpy as np
from collections import Iterable
from .qt_base_layer import QtLayerControls
from ...layers.labels._constants import Mode


class QtLabelsControls(QtLayerControls):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.mode.connect(self.set_mode)
        self.layer.events.selected_label.connect(self._on_selection_change)
        self.layer.events.brush_size.connect(self._on_brush_size_change)
        self.layer.events.contiguous.connect(self._on_contig_change)
        self.layer.events.n_dimensional.connect(self._on_n_dim_change)

        # shuffle colormap button
        self.colormapUpdate = QPushButton('shuffle colors')
        self.colormapUpdate.setObjectName('shuffle')
        self.colormapUpdate.clicked.connect(self.changeColor)
        self.colormapUpdate.setFixedHeight(28)

        # selection spinbox
        self.selectionSpinBox = QSpinBox()
        self.selectionSpinBox.setSingleStep(1)
        self.selectionSpinBox.setMinimum(0)
        self.selectionSpinBox.setMaximum(2147483647)
        self.selectionSpinBox.setValue(self.layer.selected_label)
        self.selectionSpinBox.setFixedWidth(75)
        self.selectionSpinBox.valueChanged.connect(self.changeSelection)

        sld = QSlider(Qt.Horizontal)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setMinimum(1)
        sld.setMaximum(40)
        sld.setSingleStep(1)
        value = self.layer.brush_size
        if isinstance(value, Iterable):
            if isinstance(value, list):
                value = np.asarray(value)
            value = value[:2].mean()
        sld.setValue(int(value))
        sld.valueChanged[int].connect(lambda value=sld: self.changeSize(value))
        self.brushSizeSlider = sld

        contig_cb = QCheckBox()
        contig_cb.setToolTip('contiguous editing')
        contig_cb.setChecked(self.layer.contiguous)
        contig_cb.stateChanged.connect(
            lambda state=contig_cb: self.change_contig(state)
        )
        self.contigCheckBox = contig_cb

        ndim_cb = QCheckBox()
        ndim_cb.setToolTip('n-dimensional editing')
        ndim_cb.setChecked(self.layer.n_dimensional)
        ndim_cb.stateChanged.connect(
            lambda state=ndim_cb: self.change_ndim(state)
        )
        self.ndimCheckBox = ndim_cb

        self.panzoom_button = QtModeButton(
            layer, 'zoom', Mode.PAN_ZOOM, 'Pan/zoom mode'
        )
        self.pick_button = QtModeButton(
            layer, 'picker', Mode.PICKER, 'Pick mode'
        )
        self.paint_button = QtModeButton(
            layer, 'paint', Mode.PAINT, 'Paint mode'
        )
        self.fill_button = QtModeButton(layer, 'fill', Mode.FILL, 'Fill mode')

        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.panzoom_button)
        self.button_group.addButton(self.paint_button)
        self.button_group.addButton(self.pick_button)
        self.button_group.addButton(self.fill_button)
        self.panzoom_button.setChecked(True)

        layout_option = 2
        if layout_option == 1:
            self.grid_layout.addWidget(self.panzoom_button, 0, 0)
            self.grid_layout.addWidget(self.paint_button, 0, 1)
            self.grid_layout.addWidget(self.fill_button, 0, 2)
            self.grid_layout.addWidget(self.pick_button, 0, 3)
            self.grid_layout.addWidget(QLabel('label:'), 1, 0, 1, 4)
            self.grid_layout.addWidget(self.selectionSpinBox, 2, 0, 1, 3)
            self.grid_layout.addWidget(QtColorBox(layer), 2, 3)
            self.grid_layout.addWidget(QLabel('opacity:'), 3, 0, 1, 4)
            self.grid_layout.addWidget(self.opacitySilder, 4, 0, 1, 4)
            self.grid_layout.addWidget(QLabel('brush size:'), 5, 0, 1, 4)
            self.grid_layout.addWidget(self.brushSizeSlider, 6, 0, 1, 4)
            self.grid_layout.addWidget(QLabel('blending:'), 7, 0, 1, 4)
            self.grid_layout.addWidget(self.blendComboBox, 8, 0, 1, 4)
            self.grid_layout.addWidget(QLabel('contiguous:'), 9, 0, 1, 3)
            self.grid_layout.addWidget(self.contigCheckBox, 9, 3)
            self.grid_layout.addWidget(QLabel('n-dim:'), 10, 0, 1, 3)
            self.grid_layout.addWidget(self.ndimCheckBox, 10, 3)
            self.grid_layout.addWidget(self.colormapUpdate, 11, 0, 1, 4)
            self.grid_layout.setRowStretch(12, 1)
        elif layout_option == 2:
            self.grid_layout.addWidget(self.panzoom_button, 0, 6)
            self.grid_layout.addWidget(self.paint_button, 0, 5)
            self.grid_layout.addWidget(self.fill_button, 0, 4)
            self.grid_layout.addWidget(self.pick_button, 0, 3)
            self.grid_layout.addWidget(QLabel('label:'), 1, 0, 1, 3)
            self.grid_layout.addWidget(self.selectionSpinBox, 1, 3, 1, 3)
            self.grid_layout.addWidget(QtColorBox(layer), 1, 6)
            self.grid_layout.addWidget(QLabel('opacity:'), 2, 0, 1, 3)
            self.grid_layout.addWidget(self.opacitySilder, 2, 3, 1, 4)
            self.grid_layout.addWidget(QLabel('brush size:'), 3, 0, 1, 3)
            self.grid_layout.addWidget(self.brushSizeSlider, 3, 3, 1, 4)
            self.grid_layout.addWidget(QLabel('blending:'), 4, 0, 1, 3)
            self.grid_layout.addWidget(self.blendComboBox, 4, 3, 1, 4)
            self.grid_layout.addWidget(QLabel('contiguous:'), 5, 0, 1, 3)
            self.grid_layout.addWidget(self.contigCheckBox, 5, 3)
            self.grid_layout.addWidget(QLabel('n-dim:'), 6, 0, 1, 3)
            self.grid_layout.addWidget(self.ndimCheckBox, 6, 3)
            self.grid_layout.addWidget(self.colormapUpdate, 0, 0, 1, 3)
            self.grid_layout.setRowStretch(7, 1)
            self.grid_layout.setVerticalSpacing(4)

    def mouseMoveEvent(self, event):
        self.layer.status = str(self.layer.mode)

    def set_mode(self, event):
        mode = event.mode
        if mode == Mode.PAN_ZOOM:
            self.panzoom_button.setChecked(True)
        elif mode == Mode.PICKER:
            self.pick_button.setChecked(True)
        elif mode == Mode.PAINT:
            self.paint_button.setChecked(True)
        elif mode == Mode.FILL:
            self.fill_button.setChecked(True)
        else:
            raise ValueError("Mode not recongnized")

    def changeColor(self):
        self.layer.new_colormap()

    def changeSelection(self, value):
        self.layer.selected_label = value

    def changeSize(self, value):
        self.layer.brush_size = value

    def change_contig(self, state):
        if state == Qt.Checked:
            self.layer.contiguous = True
        else:
            self.layer.contiguous = False

    def change_ndim(self, state):
        if state == Qt.Checked:
            self.layer.n_dimensional = True
        else:
            self.layer.n_dimensional = False

    def _on_selection_change(self, event):
        with self.layer.events.selected_label.blocker():
            value = self.layer.selected_label
            self.selectionSpinBox.setValue(int(value))

    def _on_brush_size_change(self, event):
        with self.layer.events.brush_size.blocker():
            value = self.layer.brush_size
            value = np.clip(int(value), 1, 40)
            self.brushSizeSlider.setValue(value)

    def _on_n_dim_change(self, event):
        with self.layer.events.n_dimensional.blocker():
            self.ndimCheckBox.setChecked(self.layer.n_dimensional)

    def _on_contig_change(self, event):
        with self.layer.events.contiguous.blocker():
            self.contigCheckBox.setChecked(self.layer.contiguous)


class QtModeButton(QRadioButton):
    def __init__(self, layer, button_name, mode, tool_tip):
        super().__init__()

        self.mode = mode
        self.layer = layer
        self.setToolTip(tool_tip)
        self.setChecked(False)
        self.setProperty('mode', button_name)
        self.toggled.connect(lambda state=self: self._set_mode(state))
        self.setFixedWidth(28)

    def _set_mode(self, bool):
        with self.layer.events.mode.blocker(self._set_mode):
            if bool:
                self.layer.mode = self.mode


class QtColorBox(QWidget):
    def __init__(self, layer):
        super().__init__()

        self.layer = layer
        self._height = 28
        self.setFixedWidth(self._height)
        self.setFixedHeight(self._height)
        self.setToolTip('Selected label color')

        self.layer.events.selected_label.connect(self.update_color)

    def update_color(self, event):
        self.update()

    def paintEvent(self, event):
        """Paint the colorbox.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        painter = QPainter(self)
        if self.layer._selected_color is None:
            painter.setPen(QColor(230, 230, 230))
            painter.setBrush(QColor(230, 230, 230))
            for i in range(self._height // 6 + 1):
                for j in range(self._height // 6 + 1):
                    if (i % 2 == 0 and j % 2 == 0) or (
                        i % 2 == 1 and j % 2 == 1
                    ):
                        painter.drawRect(i * 6, j * 6, 5, 5)
        else:
            color = 255 * self.layer._selected_color
            color = color.astype(int)
            painter.setPen(QColor(*list(color)))
            painter.setBrush(QColor(*list(color)))
            painter.drawRect(0, 0, self._height, self._height)
