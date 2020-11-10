"""QtOctreeInfo and QtOctreeInfoLayout classes.
"""
from typing import Callable

import numpy as np
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ....layers.image.experimental.octree_image import OctreeImage

IntCallback = Callable[[int], None]


class QtSimpleTable(QTableWidget):
    """A table of keys and values."""

    def __init__(self):
        super().__init__()
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.resizeRowsToContents()
        self.setShowGrid(False)

    def set_values(self, values: dict) -> None:
        """Populate the table with keys and values.

        values : dict
            Populate with these keys and values.
        """
        self.setRowCount(len(values))
        self.setColumnCount(2)
        for i, (key, value) in enumerate(values.items()):
            self.setItem(i, 0, QTableWidgetItem(key))
            self.setItem(i, 1, QTableWidgetItem(value))


class QtLevelCombo(QHBoxLayout):
    """Combo box to choose an octree level or AUTO.

    Parameters
    ----------
    num_levels : int
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

    def __init__(
        self,
        layer: OctreeImage,
        on_set_level: IntCallback,
        on_set_track: IntCallback,
    ):
        super().__init__()

        self.level = QtLevelCombo(layer.num_octree_levels, on_set_level)
        self.addLayout(self.level)

        self.track = QCheckBox("Track View")
        self.track.stateChanged.connect(on_set_track)
        self.track.setChecked(layer.track_view)
        self.addWidget(self.track)

        self.table = QtSimpleTable()
        self.addWidget(self.table)

        self.set_layout(layer)  # Initial settings.

    def set_layout(self, layer: OctreeImage):
        """Set controls based on the layer.

        Parameters
        ----------
        layer : OctreeImage
            Set controls based on this layer.
        """
        self.level.set_index(0 if layer.auto_level else layer.octree_level + 1)
        self.table.set_values(self._get_values(layer))

    def _get_values(self, layer: OctreeImage) -> None:
        """Set the table based on the layer.

        layer : OctreeImage
            Set values from this layer.
        """

        def _str(shape) -> str:
            return f"{shape[1]}x{shape[0]}"

        level_info = layer.octree_level_info

        shape_in_tiles = level_info.shape_in_tiles
        shape_in_tiles_str = _str(shape_in_tiles)
        num_tiles = shape_in_tiles[0] * shape_in_tiles[1]

        return {
            "Level": f"{layer.octree_level}",
            "Tiles": f"{shape_in_tiles_str} = {num_tiles}",
            "Tile Shape": _str([layer.tile_size, layer.tile_size]),
            "Layer Shape": _str(level_info.image_shape),
        }


class QtOctreeInfo(QFrame):
    """Frame showing the octree level and tile size.

    layer : OctreeImage
        Show info about this layer.
    """

    def __init__(self, layer: OctreeImage):
        super().__init__()
        self.layer = layer

        self.layout = QtOctreeInfoLayout(
            layer, self._on_set_level, self._on_set_track
        )
        self.setLayout(self.layout)

        # Initial update and connect for future updates.
        self._set_layout()
        layer.events.auto_level.connect(self._set_layout)
        layer.events.octree_level.connect(self._set_layout)
        layer.events.tile_size.connect(self._set_layout)

    def _set_layout(self, event=None):
        """Set layout controls based on the layer."""
        self.layout.set_layout(self.layer)

    def _on_set_level(self, value: int) -> None:
        """Set octree level in the layer.

        Parameters
        ----------
        value : int
            The new level index.
        """
        # Drop down has AUTO at index 0.
        if value == 0:
            self.layer.auto_level = True
        else:
            self.layer.auto_level = False
            self.layer.octree_level = value - 1

    def _on_set_track(self, value: int) -> None:
        """Set whether rendering should track the current value.

        value : int
            If non-zero then rendering track the view as it moves.
        """
        self.layer.track_view = value != 0
