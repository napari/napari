"""QtOctreeInfo class.
"""
from typing import Callable

import numpy as np
from qtpy.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel, QVBoxLayout


class QtOctreeLevelCombo(QHBoxLayout):
    def __init__(self, layer, update_layer):
        super().__init__()

        self.addWidget(QLabel("Octree Level"))

        current_level = layer.octree_level
        levels = [str(x) for x in np.arange(0, layer.num_octree_levels)]

        self.octree_level = QComboBox()
        self.octree_level.addItems(levels)
        self.octree_level.activated[int].connect(update_layer)
        self.octree_level.setCurrentIndex(current_level)
        self.addWidget(self.octree_level)

    def set_level(self, level):
        self.octree_level.setCurrentIndex(level)


class QtOctreeInfoLayout(QVBoxLayout):
    """Layout of the octree info frame.

    Parameters
    ----------
    layer : Layer
        Show octree info for this layer
    on_new_octree_level : Callable[[int], None]
        Call this when the octree level is changed.
    """

    def __init__(self, layer, update_layer: Callable[[int], None]):
        super().__init__()

        self.octree_level = QtOctreeLevelCombo(layer, update_layer)
        self.addLayout(self.octree_level)

        self.tile_size = QLabel()
        self.addWidget(self.tile_size)

        self.update(layer)

    def update(self, layer):
        """Update the layout with latest information for the layer.

        Parameters
        ----------
        layer : Layer
            Update with information from this layer.
        """
        self.octree_level.set_level(layer.octree_level)

        size = layer.tile_size
        self.tile_size.setText(f"Tile Size: {size}x{size}")


class QtOctreeInfo(QFrame):
    """Frame showing the octree level and tile size.

    layer : Layer
        Show info about this layer.
    """

    def __init__(self, layer):
        super().__init__()

        def _update_layer(value):
            layer.octree_level = value

        self.layout = QtOctreeInfoLayout(layer, _update_layer)
        self.setLayout(self.layout)

        def _update_layout(event=None):
            if layer.octree_level is not None:
                self.layout.update(layer)

        # Initial update and connect for future updates.
        _update_layout()
        layer.events.octree_level.connect(_update_layout)
        layer.events.tile_size.connect(_update_layout)
