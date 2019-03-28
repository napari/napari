from PyQt5.QtWidgets import QPushButton
from ..._base_layer import QtLayer


class QtLabelsLayer(QtLayer):
    def __init__(self, layer):
        super().__init__(layer)

        self.layer.events.colormap.connect(self._on_colormap_change)

        self.colormap_update = QPushButton('shuffle colors')
        self.colormap_update.clicked.connect(self.changeColor)
        self.grid_layout.addWidget(self.colormap_update, 3, 0, 1, 2)

        self.setExpanded(False)

    def changeColor(self):
        self.layer.new_colormap()

    def _on_colormap_change(self, event):
        self.layer._node.cmap = self.layer.colormap
