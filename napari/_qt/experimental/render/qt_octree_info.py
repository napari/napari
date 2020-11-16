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

        # Checkbox to toggle the ChunkCache.
        self._create_checkbox(
            "Chunk Cache", chunk_loader.cache.enabled, on_set_cache
        )

        def on_set_track(value: int):
            self.layer.track_view = value != 0

        # Checkbox to toggle if the drawn tiles should update as the view
        # moves around. They normal state is yes.
        self.track = QCheckBox("Track View")
        self.track.stateChanged.connect(on_set_track)
        self.track.setChecked(layer.track_view)
        self.addWidget(self.track)

        # Checkbox to toggle debug grid around the tiles.
        self.track = QCheckBox("Show Grid")
        self.track.stateChanged.connect(on_set_grid)
        self.track.setChecked(layer.show_grid)
        self.addWidget(self.track)

        # Choose AUTO or which octree level to view.
        self.level = QtLevelCombo(layer.num_octree_levels, on_set_level)
        self.addLayout(self.level)

        # Keys and values about the octree.
        self.table = QtSimpleTable()
        self.addWidget(self.table)

        self.set_layout(layer)  # Initial settings.

    def _create_checkbox(self, label, initial_value, callback):
        checkbox = QCheckBox(label)
        checkbox.stateChanged.connect(callback)
        checkbox.setChecked(initial_value)
        self.addWidget(checkbox)

    def set_layout(self, layer: OctreeImage):
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
        self._set_layout()
        layer.events.auto_level.connect(self._set_layout)
        layer.events.octree_level.connect(self._set_layout)
        layer.events.tile_size.connect(self._set_layout)

    def _set_layout(self, _event=None):
        """Set layout controls based on the layer."""
        self.layout.set_layout(self.layer)
