from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QComboBox
from ..._base_layer import QtLayer


class QtImageLayer(QtLayer):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.colormap.connect(self._on_colormap_change)
        self.layer.events.interpolation.connect(self._on_interpolation_change)

        comboBox = QComboBox()
        for cmap in self.layer.colormaps:
            comboBox.addItem(cmap)
        comboBox._allitems = set(self.layer.colormaps)
        index = comboBox.findText(self.layer.colormap_name, Qt.MatchFixedString)
        comboBox.setCurrentIndex(index)
        comboBox.activated[str].connect(
            lambda text=comboBox: self.changeColor(text))
        self.grid_layout.addWidget(QLabel('colormap:'), 3, 0)
        self.grid_layout.addWidget(comboBox, 3, 1)
        self.colormap_combobox = comboBox

        interp_comboBox = QComboBox()
        for interp in self.layer._interpolation_names:
            interp_comboBox.addItem(interp)
        index = interp_comboBox.findText(
            self.layer.interpolation, Qt.MatchFixedString)
        interp_comboBox.setCurrentIndex(index)
        interp_comboBox.activated[str].connect(
            lambda text=interp_comboBox: self.changeInterpolation(text))
        self.interpComboBox = interp_comboBox
        self.grid_layout.addWidget(QLabel('interpolation:'), 4, 0)
        self.grid_layout.addWidget(interp_comboBox, 4, 1)

        self.setExpanded(False)

    def changeColor(self, text):
        self.layer.colormap = text

    def changeInterpolation(self, text):
        self.layer.interpolation = text

    def _on_interpolation_change(self, event):
        with self.layer.events.interpolation.blocker():
            index = self.interpComboBox.findText(
                self.layer.interpolation, Qt.MatchFixedString)
            self.interpComboBox.setCurrentIndex(index)

    def _on_colormap_change(self, event):
        name = self.layer.colormap_name
        if name not in self.colormap_combobox._allitems:
            self.colormap_combobox._allitems.add(name)
            self.colormap_combobox.addItem(name)
        if name != self.colormap_combobox.currentText():
            self.colormap_combobox.setCurrentText(name)
