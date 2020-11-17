"""QtOctreeInfo and QtOctreeInfoLayout classes.

Shows octree-specific information in the QtRender widget.
"""
from typing import Callable

import numpy as np
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)

from ....components.experimental import chunk_loader
from ....layers.image.experimental.octree_image import OctreeImage
from .qt_render_widgets import QtSimpleTable

IntCallback = Callable[[int], None]


def _get_table_values(layer: OctreeImage) -> dict:
    """Get keys/values about this octree image for the table.

    layer : OctreeImage
        Get values for this octree image.
    """

    def _shape(shape) -> str:
        return f"{shape[1]}x{shape[0]}"

    level_info = layer.octree_level_info

    if level_info is None:
        return {}

    shape_in_tiles = level_info.shape_in_tiles
    num_tiles = shape_in_tiles[0] * shape_in_tiles[1]

    return {
        "Level": f"{layer.octree_level}",
        "Tiles": f"{_shape(shape_in_tiles)} = {num_tiles}",
        "Tile Shape": _shape([layer.tile_size, layer.tile_size]),
        "Layer Shape": _shape(level_info.image_shape),
    }


class QtLevelCombo(QHBoxLayout):
    """Combo box to choose an octree level or AUTO.

    Parameters
    ----------
    num_levels : int
        The number of available levels.
    on
    """

    def __init__(self, num_levels: int, on_set_level: IntCallback):
        super().__init__()

        self.addWidget(QLabel("Octree Level"))

        # AUTO means napari selects the appropriate octree level
        # dynamically as you zoom in or out.
        items = ["AUTO"] + [str(x) for x in np.arange(0, num_levels)]

        self.level = QComboBox()
        self.level.addItems(items)
        self.level.activated[int].connect(on_set_level)
        self.addWidget(self.level)

    def set_index(self, index: int) -> None:
        """Set the dropdown's value.

        Parameters
        ----------
        index : int
            Index of dropdown where AUTO is index 0.
        """
        # Add one because AUTO is at index 0.
        self.level.setCurrentIndex(0 if index is None else (index + 1))


class QtOctreeInfoLayout(QVBoxLayout):
    """OctreeImage specific information.

    Combo base to choose octree layer or set to AUTO for the normal rendering
    mode where the correct level is chosen automatically. (not working yet)

    Parameters
    ----------
    layer : OctreeImage
        Show octree info for this layer.
    on_set_level : IntCallback
        Call this when the octree level is changed.
    """

    def __init__(self, layer: OctreeImage):
        super().__init__()

        self.layer = layer

        def on_set_cache(value: int) -> None:
            chunk_loader.cache.enabled = value != 0

        def on_set_grid(value: int) -> None:
            self.layer.show_grid = value != 0

        def on_set_level(value: int) -> None:
            # Drop down has AUTO at index 0.
            if value == 0:
                self.layer.auto_level = True
            else:
                self.layer.auto_level = False
                self.layer.octree_level = value - 1

        def on_set_track(value: int):
            self.layer.track_view = value != 0

        # Toggle the ChunkCache.
        cache_enabled = chunk_loader.cache.enabled
        self._create_checkbox("Chunk Cache", cache_enabled, on_set_cache)

        # Toggle tracking: drawn tiles track view as it moves.
        self._create_checkbox("Track View", layer.track_view, on_set_track)

        # Toggle debug grid drawn around tiles.
        self._create_checkbox("Show Grid", layer.show_grid, on_set_grid)

        # Select which octree level to view or AUTO for normal mode.
        self.level = QtLevelCombo(layer.num_octree_levels, on_set_level)
        self.addLayout(self.level)

        # Show some keys and values about the octree.
        self.table = QtSimpleTable()
        self.addWidget(self.table)

        self.set_controls(layer)  # Initial settings.

    def _create_checkbox(self, label, initial_value, callback):
        checkbox = QCheckBox(label)
        checkbox.stateChanged.connect(callback)
        checkbox.setChecked(initial_value)
        self.addWidget(checkbox)

    def set_controls(self, layer: OctreeImage):
        """Set controls based on the layer.

        Parameters
        ----------
        layer : OctreeImage
            Set controls based on this layer.
        """
        self.level.set_index(0 if layer.auto_level else layer.octree_level + 1)
        self.table.set_values(_get_table_values(layer))


class QtOctreeInfo(QFrame):
    """Frame showing the octree level and tile size.

    layer : OctreeImage
        Show info about this layer.
    """

    def __init__(self, layer: OctreeImage):
        super().__init__()
        self.layer = layer

        self.layout = QtOctreeInfoLayout(layer)
        self.setLayout(self.layout)

        # Initial update and connect for future updates.
        self._update()  # initial update
        layer.events.auto_level.connect(self._update)
        layer.events.octree_level.connect(self._update)
        layer.events.tile_size.connect(self._update)

    def _update(self, _event=None):
        """Set controls based on the current layer setting."""
        self.layout.set_controls(self.layer)
