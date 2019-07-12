from .qt_base_layer import QtLayerControls, QtLayerProperties
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QComboBox

from ..layers.volume._constants import Rendering


class QtVolumeControls(QtLayerControls):
    def __init__(self, layer):
        super().__init__(layer)


class QtVolumeProperties(QtLayerProperties):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.colormap.connect(self._on_colormap_change)
        self.layer.events.rendering.connect(self._on_rendering_change)

        # TODO Remove the default blending widget on QtLayerProperties
        # as it corresponds to blending of layers in RGB and RGBA image
        self.grid_layout.removeWidget(self.blendComboBox)
        self.blendComboBox.deleteLater()
        self.blendComboBox.setParent(None)

        row = self.grid_layout.rowCount() - 1
        renderComboBox = QComboBox()
        for render in Rendering:
            renderComboBox.addItem(str(render))
        index = renderComboBox.findText(
            self.layer.rendering, Qt.MatchFixedString
        )
        renderComboBox.setCurrentIndex(index)
        renderComboBox.activated[str].connect(
            lambda text=renderComboBox: self.changeRendering(text)
        )
        self.renderComboBox = renderComboBox
        self.grid_layout.addWidget(QLabel('rendering:'), row, self.name_column)
        self.grid_layout.addWidget(renderComboBox, row, self.property_column)

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

    def changeRendering(self, text):
        self.layer.rendering = text

    def _on_rendering_change(self, event):
        with self.layer.events.rendering.blocker():
            index = self.renderComboBox.findText(
                self.layer.rendering, Qt.MatchFixedString
            )
            self.renderComboBox.setCurrentIndex(index)
