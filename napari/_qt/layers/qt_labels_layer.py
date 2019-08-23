from collections import Iterable

import numpy as np
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

from ...layers import Labels
from ...layers.labels._constants import Mode
from .qt_base_layer import QtLayerControls, QtLayerProperties


class QtLabelsControls(QtLayerControls, layer=Labels):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.mode.connect(self.set_mode)

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

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 20, 10, 10)
        layout.addWidget(self.panzoom_button)
        layout.addWidget(self.paint_button)
        layout.addWidget(self.pick_button)
        layout.addWidget(self.fill_button)
        layout.addWidget(QtColorBox(layer))
        layout.addStretch(0)
        self.setLayout(layout)
        self.setMouseTracking(True)

        self.panzoom_button.setChecked(True)

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


class QtLabelsProperties(QtLayerProperties, layer=Labels):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.selected_label.connect(self._on_selection_change)
        self.layer.events.brush_size.connect(self._on_brush_size_change)
        self.layer.events.contiguous.connect(self._on_contig_change)
        self.layer.events.n_dimensional.connect(self._on_n_dim_change)

        self.colormap_update = QPushButton('click')
        self.colormap_update.setObjectName('shuffle')
        self.colormap_update.clicked.connect(self.changeColor)
        self.colormap_update.setFixedWidth(112)
        self.colormap_update.setFixedHeight(25)
        shuffle_label = QLabel('shuffle colors:')
        shuffle_label.setObjectName('shuffle-label')
        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(shuffle_label, row, self.name_column)
        self.grid_layout.addWidget(
            self.colormap_update, row, self.property_column
        )

        # selection spinbox
        self.selection_spinbox = QSpinBox()
        self.selection_spinbox.setSingleStep(1)
        self.selection_spinbox.setMinimum(0)
        self.selection_spinbox.setMaximum(2147483647)
        self.selection_spinbox.setValue(self.layer.selected_label)
        self.selection_spinbox.setFixedWidth(75)
        self.selection_spinbox.valueChanged.connect(self.changeSelection)
        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(QLabel('label:'), row, self.name_column)
        self.grid_layout.addWidget(
            self.selection_spinbox, row, self.property_column
        )

        sld = QSlider(Qt.Horizontal, self)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.setFixedWidth(110)
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
        self.brush_size_slider = sld
        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(
            QLabel('brush size:'), row, self.name_column
        )
        self.grid_layout.addWidget(sld, row, self.property_column)

        contig_cb = QCheckBox()
        contig_cb.setToolTip('contiguous editing')
        contig_cb.setChecked(self.layer.contiguous)
        contig_cb.stateChanged.connect(
            lambda state=contig_cb: self.change_contig(state)
        )
        self.contig_checkbox = contig_cb
        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(
            QLabel('contiguous:'), row, self.name_column
        )
        self.grid_layout.addWidget(contig_cb, row, self.property_column)

        ndim_cb = QCheckBox()
        ndim_cb.setToolTip('n-dimensional editing')
        ndim_cb.setChecked(self.layer.n_dimensional)
        ndim_cb.stateChanged.connect(
            lambda state=ndim_cb: self.change_ndim(state)
        )
        self.ndim_checkbox = ndim_cb
        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(QLabel('n-dim:'), row, self.name_column)
        self.grid_layout.addWidget(ndim_cb, row, self.property_column)

        self.setExpanded(False)

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
            self.selection_spinbox.setValue(int(value))

    def _on_brush_size_change(self, event):
        with self.layer.events.brush_size.blocker():
            value = self.layer.brush_size
            value = np.clip(int(value), 1, 40)
            self.brush_size_slider.setValue(value)

    def _on_n_dim_change(self, event):
        with self.layer.events.n_dimensional.blocker():
            self.ndim_checkbox.setChecked(self.layer.n_dimensional)

    def _on_contig_change(self, event):
        with self.layer.events.contiguous.blocker():
            self.contig_checkbox.setChecked(self.layer.contiguous)
