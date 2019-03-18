from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QComboBox, QPushButton
from ..._base_layer import QtLayer


class QtImageLayer(QtLayer):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.colormap.connect(self._on_colormap_change)
        self.layer.events.interpolation.connect(self._on_interpolation_change)

        self.colormap_update = QPushButton('shuffle colors', parent=self)
        self.colormap_update.clicked.connect(self.changeColor)

        self.setExpanded(False)

    def changeColor(self):
        self.layer.new_colormap()

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
