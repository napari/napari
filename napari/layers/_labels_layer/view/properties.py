from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QPushButton,
    QComboBox,
    QSlider,
    QCheckBox,
    QLabel,
    QSpinBox,
)
import numpy as np
from collections import Iterable
from ..._base_layer import QtLayerProperties


class QtLabelsProperties(QtLayerProperties):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.colormap.connect(self._on_colormap_change)
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

    def _on_colormap_change(self, event):
        self.layer._node.cmap = self.layer.colormap

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
