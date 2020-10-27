"""QtOctreeInfo class.
"""
from typing import Callable

import numpy as np
from qtpy.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

AUTO_INDEX = 0


def _index_to_level(index):
    return index - 1  # Since AUTO is index 0


def _level_to_index(level):
    return level + 1  # Since AUTO is index 0


class QtLevelCombo(QHBoxLayout):
    def __init__(self, num_levels, set_level):
        super().__init__()

        self.addWidget(QLabel("Octree Level"))

        levels = [str(x) for x in np.arange(0, num_levels)]
        items = ["AUTO"] + levels

        self.level = QComboBox()
        self.level.addItems(items)
        self.level.activated[int].connect(set_level)
        self.addWidget(self.level)

    def set_index(self, index):
        self.level.setCurrentIndex(index)


class QtOctreeInfoLayout(QVBoxLayout):
    """Layout of the octree info frame.

    Parameters
    ----------
    layer : Layer
        Show octree info for this layer
    on_new_octree_level : Callable[[int], None]
        Call this when the octree level is changed.
    """

    def __init__(
        self, layer, set_level: Callable[[int], None],
    ):
        super().__init__()

        self.level = QtLevelCombo(layer.num_octree_levels, set_level)
        self.addLayout(self.level)

        self.table = QTableWidget()
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.resizeRowsToContents()
        self.table.setShowGrid(False)
        self.addWidget(self.table)

        self.set_layout(layer)  # Initial settings.

    def set_layout(self, layer):
        """Set controls based on the layer.

        Parameters
        ----------
        layer : Layer
            Set controls based on this layer.
        """
        if layer.auto_level:
            self.level.set_index(AUTO_INDEX)
        else:
            self.level.set_index(_level_to_index(layer.octree_level))

        level_info = layer.octree_level_info

        def _str(shape) -> str:
            return f"{shape[1]}x{shape[0]}"

        values = {
            "Level": str(layer.octree_level),
            "Tile Shape": _str(level_info.tile_shape),
            "Tile Size": _str([layer.tile_size, layer.tile_size]),
            "Layer Shape": _str(level_info.image_shape),
        }
        self.table.setRowCount(len(values))
        self.table.setColumnCount(2)
        for i, (key, value) in enumerate(values.items()):
            self.table.setItem(i, 0, QTableWidgetItem(key))
            self.table.setItem(i, 1, QTableWidgetItem(value))


class QtOctreeInfo(QFrame):
    """Frame showing the octree level and tile size.

    layer : Layer
        Show info about this layer.
    """

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.layout = QtOctreeInfoLayout(layer, self._set_level)
        self.setLayout(self.layout)

        # Initial update and connect for future updates.
        self._set_layout()
        layer.events.auto_level.connect(self._set_layout)
        layer.events.octree_level.connect(self._set_layout)
        layer.events.tile_size.connect(self._set_layout)

    def _set_layout(self, event=None):
        self.layout.set_layout(self.layer)

    def _set_level(self, value):
        if value == AUTO_INDEX:
            self.layer.auto_level = True
        else:
            self.layer.auto_level = False
            self.layer.octree_level = _index_to_level(value)
