from qtpy.QtWidgets import QHBoxLayout
from .qt_base_layer import QtLayerProperties
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QComboBox


class QtImageBaseProperties(QtLayerProperties):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.colormap.connect(self._on_colormap_change)
        row = self.grid_layout.rowCount()
        comboBox = QComboBox()
        for cmap in self.layer.colormaps:
            comboBox.addItem(cmap)
        comboBox._allitems = set(self.layer.colormaps)
        index = comboBox.findText(self.layer.colormap[0], Qt.MatchFixedString)
        comboBox.setCurrentIndex(index)
        comboBox.activated[str].connect(
            lambda text=comboBox: self.changeColor(text)
        )
        self.grid_layout.addWidget(QLabel('colormap:'), row, self.name_column)
        self.grid_layout.addWidget(comboBox, row, self.property_column)
        self.colormap_combobox = comboBox

        self.setExpanded(False)

    def changeColor(self, text):
        self.layer.colormap = text

    def _on_colormap_change(self, event):
        name = self.layer.colormap[0]
        if name not in self.colormap_combobox._allitems:
            self.colormap_combobox._allitems.add(name)
            self.colormap_combobox.addItem(name)
        if name != self.colormap_combobox.currentText():
            self.colormap_combobox.setCurrentText(name)
