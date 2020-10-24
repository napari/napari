"""QtImageInfo class.
"""
from typing import Callable

from qtpy.QtWidgets import QFrame, QLabel, QVBoxLayout

from .qt_labeled_spin_box import QtLabeledSpinBox


class QtImageInfoLayout(QVBoxLayout):
    """Layout of the image info frame.

    Parameters
    ----------
    layer : Layer
        The layer we are associated with.
    """

    def __init__(self, layer):
        super().__init__()

        shape = layer.data.shape
        height, width = shape[1:3]  # Which dims are really width/height?

        # Dimension related labels.
        self.addWidget(QLabel(f"Shape: {shape}"))
        self.addWidget(QLabel(f"Width: {width}"))
        self.addWidget(QLabel(f"Height: {height}"))


class QtImageInfo(QFrame):
    """Frame showing image shape and dimensions.

    layer : Layer
        Show info about this layer.
    """

    def __init__(self, layer):
        super().__init__()

        layout = QtImageInfoLayout(layer)
        self.setLayout(layout)


class QtOctreeInfoLayout(QVBoxLayout):
    """Layout of the octree info frame.

    Parameters
    ----------
    layer : Layer
        The layer we are associated with.
    on_new_octree_level : Callable[[int], None]
        Call this when the octree level is changed.
    """

    def __init__(self, layer, on_new_octree_level: Callable[[int], None]):
        super().__init__()

        # SpinBox showing the layer's current octree level.
        max_level = layer.num_octree_levels - 1
        self.octree_level = QtLabeledSpinBox(
            "Octree Level",
            0,
            range(0, max_level, 1),
            connect=on_new_octree_level,
        )
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
        self.octree_level.spin.setValue(layer.octree_level)

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

        # Update layout now and hook to event for future updates.
        _update_layout()
        layer.events.octree_level.connect(_update_layout)
        layer.events.tile_size.connect(_update_layout)
